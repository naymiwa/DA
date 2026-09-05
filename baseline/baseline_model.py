"""
Baseline model untuk kompetisi DAC IFest 2026 - Kesesuaian Judul & Isi Berita.

Pendekatan:
  1. Fitur handcrafted (overlap kata, overlap angka, cosine similarity TF-IDF
     antara judul & isi, dsb) -> menangkap intuisi "judul vs isi".
  2. Fitur TF-IDF (word + char n-gram) dari teks gabungan "judul [SEP] isi"
     -> menangkap sinyal leksikal tambahan.
  3. Model: Logistic Regression dengan class_weight='balanced' karena label
     sangat imbalance (90% Sesuai vs 10% Tidak Sesuai).
  4. Validasi: Stratified hold-out 80/20, metrik Macro F1 (sama seperti
     leaderboard). Untuk submission final disarankan pakai Stratified
     K-Fold (mis. 5-fold) agar skor lebih stabil -- lihat catatan di bawah.
  5. Threshold tuning: mencari ambang batas probabilitas yang memaksimalkan
     Macro F1 di data validasi (bukan asal 0.5), karena data imbalance.
  6. Retrain di seluruh data train, prediksi test, simpan submission.csv.

WAJIB: set semua random_state / seed (aturan lomba: notebook harus
menyertakan seed).

Cara pakai:
    python3 baseline/baseline_model.py

Catatan: skrip ini adalah BASELINE / titik awal, bukan solusi akhir.
Silakan dikembangkan (lihat bagian "Ide pengembangan" di bawah & di panduan).
"""

import os
import re
import glob
import zipfile
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data/penyisihan-dac-ifest-2026"
OUT_PATH = "baseline/baseline_submission.csv"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
if not os.path.exists(f"{DATA_DIR}/train.csv"):
    zips = glob.glob("*.zip") + glob.glob("data/*.zip")
    if not zips:
        raise FileNotFoundError(
            "train.csv tidak ditemukan dan tidak ada file .zip dataset Kaggle "
            "di root repo. Unduh dataset dari Kaggle lalu taruh zip-nya di "
            "root project, atau extract manual ke folder data/."
        )
    print(f"Mengekstrak dataset dari {zips[0]} ...")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall("data")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"train: {train.shape}, test: {test.shape}, "
      f"sample_submission: {sample_sub.shape}")

# PENTING: Kaggle menolak submission kalau jumlah baris tidak sama persis
# dengan sample_submission. Kalau angka di bawah ini berbeda, berarti
# dataset lokal SUDAH KADALUARSA -> download ulang test.csv &
# sample_submission.csv terbaru dari tab Data di Kaggle.
if len(test) != len(sample_sub):
    print("\n[!!] PERINGATAN: jumlah baris test.csv "
          f"({len(test)}) != sample_submission.csv ({len(sample_sub)}).")
    print("[!!] Dataset lokal kemungkinan versi lama. Download ulang "
          "dataset TERBARU dari Kaggle sebelum submit.\n")


# ---------------------------------------------------------------------------
# 2. Handcrafted features (judul vs isi)
# ---------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def extract_numbers(text):
    return set(re.findall(r"\d+", str(text)))


def make_features(df):
    feats = pd.DataFrame(index=df.index)

    title_tokens = df["title"].apply(tokenize)
    content_tokens = df["content"].apply(tokenize)

    title_set = title_tokens.apply(set)
    content_set = content_tokens.apply(set)

    inter = [len(t & c) for t, c in zip(title_set, content_set)]
    union = [len(t | c) for t, c in zip(title_set, content_set)]

    feats["title_len"] = title_tokens.apply(len)
    feats["content_len"] = content_tokens.apply(len)
    feats["len_ratio"] = feats["title_len"] / (feats["content_len"] + 1)

    # overlap kata judul yang muncul di isi (recall dari sisi judul)
    feats["word_overlap_ratio"] = [
        i / len(t) if len(t) > 0 else 0.0 for i, t in zip(inter, title_set)
    ]
    # jaccard similarity keseluruhan
    feats["jaccard"] = [i / u if u > 0 else 0.0 for i, u in zip(inter, union)]

    # overlap angka (tanggal, jumlah korban, nominal, dsb sering jadi
    # penentu utama "beda peristiwa")
    title_nums = df["title"].apply(extract_numbers)
    content_nums = df["content"].apply(extract_numbers)
    feats["num_in_title"] = title_nums.apply(len)
    feats["num_overlap_ratio"] = [
        len(tn & cn) / len(tn) if len(tn) > 0 else 1.0  # tidak ada angka = netral
        for tn, cn in zip(title_nums, content_nums)
    ]

    # overlap kata berkapital di awal (proxy nama orang/lembaga/tempat,
    # tanpa perlu library NER tambahan)
    def capitalized_words(text):
        return set(w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", str(text)))

    title_caps = df["title"].apply(capitalized_words)
    content_caps = df["content"].apply(capitalized_words)
    feats["cap_overlap_ratio"] = [
        len(tc & cc) / len(tc) if len(tc) > 0 else 1.0
        for tc, cc in zip(title_caps, content_caps)
    ]

    return feats


print("Membuat handcrafted features...")
train_feats = make_features(train)
test_feats = make_features(test)

# skala fitur handcrafted (mis. content_len bisa ribuan, sedangkan
# overlap_ratio 0-1) supaya Logistic Regression konvergen lebih cepat &
# stabil ketika digabung dengan fitur TF-IDF yang sudah ternormalisasi
scaler = StandardScaler()
train_feats_scaled = scaler.fit_transform(train_feats.values)
test_feats_scaled = scaler.transform(test_feats.values)

# ---------------------------------------------------------------------------
# 3. TF-IDF cosine similarity judul vs isi (fit di kalimat gabungan)
# ---------------------------------------------------------------------------
print("Fitting TF-IDF untuk cosine similarity judul-isi...")
sim_vectorizer = TfidfVectorizer(
    max_features=8000, ngram_range=(1, 1), min_df=5
)
all_text_for_sim = pd.concat([train["title"], train["content"],
                               test["title"], test["content"]])
sim_vectorizer.fit(all_text_for_sim)


def cosine_sim_batch(titles, contents, vectorizer):
    t_vecs = vectorizer.transform(titles)
    c_vecs = vectorizer.transform(contents)
    # normalisasi lalu dot product (karena TF-IDF sudah otomatis
    # ternormalisasi dgn norm='l2' default, dot product = cosine sim)
    sims = np.asarray(t_vecs.multiply(c_vecs).sum(axis=1)).ravel()
    return sims


train_feats["tfidf_cosine_sim"] = cosine_sim_batch(
    train["title"], train["content"], sim_vectorizer
)
test_feats["tfidf_cosine_sim"] = cosine_sim_batch(
    test["title"], test["content"], sim_vectorizer
)

# ---------------------------------------------------------------------------
# 4. TF-IDF fitur leksikal tambahan dari teks gabungan "judul [SEP] isi"
# ---------------------------------------------------------------------------
print("Fitting TF-IDF gabungan judul+isi (word-level)...")
train_combined = train["title"] + " [SEP] " + train["content"]
test_combined = test["title"] + " [SEP] " + test["content"]

word_vectorizer = TfidfVectorizer(
    max_features=8000, ngram_range=(1, 1), min_df=8, sublinear_tf=True
)
X_train_tfidf = word_vectorizer.fit_transform(train_combined)
X_test_tfidf = word_vectorizer.transform(test_combined)

# ---------------------------------------------------------------------------
# 5. Gabungkan semua fitur
# ---------------------------------------------------------------------------
X_train = hstack([csr_matrix(train_feats_scaled), X_train_tfidf]).tocsr()
X_test = hstack([csr_matrix(test_feats_scaled), X_test_tfidf]).tocsr()
y_train = train["label"].values

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# 6. Validasi (stratified hold-out 80/20 -- cepat untuk baseline) +
#    threshold tuning
#
#    NB: untuk laporan/submission FINAL, ganti bagian ini dengan
#    StratifiedKFold(n_splits=5) (sudah di-import) supaya skor Macro F1
#    yang dilaporkan di makalah lebih stabil/tidak bergantung 1 split saja.
#    Kode di sini sengaja pakai 1 split agar baseline cepat dijalankan.
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

tr_idx, val_idx = train_test_split(
    np.arange(len(train)), test_size=0.2, stratify=y_train, random_state=SEED
)
X_tr, X_val = X_train[tr_idx], X_train[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]

print("\nTraining Logistic Regression (hold-out 80/20)...")
clf = LogisticRegression(
    max_iter=200,
    solver="liblinear",  # cepat & stabil untuk data sparse + binary
    class_weight="balanced",  # penting karena imbalance 90/10
    C=1.0,
    random_state=SEED,
)
clf.fit(X_tr, y_tr)

val_proba = clf.predict_proba(X_val)[:, 1]
val_pred_default = (val_proba >= 0.5).astype(int)
f1_default = f1_score(y_val, val_pred_default, average="macro")
print(f"Macro F1 di validasi (threshold=0.5): {f1_default:.4f}")

# Cari threshold terbaik berdasarkan prediksi validasi
print("\nMencari threshold optimal untuk Macro F1...")
best_threshold, best_f1 = 0.5, 0.0
for th in np.arange(0.05, 0.96, 0.01):
    pred = (val_proba >= th).astype(int)
    f1 = f1_score(y_val, pred, average="macro")
    if f1 > best_f1:
        best_f1, best_threshold = f1, th

print(f"Threshold optimal: {best_threshold:.2f} -> Macro F1 validasi: {best_f1:.4f}")
print("\nClassification report (validasi, threshold optimal):")
print(classification_report(
    y_val, (val_proba >= best_threshold).astype(int),
    target_names=["Tidak Sesuai (0)", "Sesuai (1)"]
))

# ---------------------------------------------------------------------------
# 7. Retrain di seluruh data train, prediksi test set
# ---------------------------------------------------------------------------
print("\nTraining model final di seluruh data train...")
final_clf = LogisticRegression(
    max_iter=300, solver="liblinear", class_weight="balanced", C=1.0,
    random_state=SEED,
)
final_clf.fit(X_train, y_train)

test_proba = final_clf.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print(f"Distribusi prediksi test -> label 0: {(test_pred == 0).sum()}, "
      f"label 1: {(test_pred == 1).sum()}")

# ---------------------------------------------------------------------------
# 8. Simpan submission.csv
# ---------------------------------------------------------------------------
# Bangun submission mengikuti KOLOM id dari sample_submission supaya jumlah
# baris & urutannya PERSIS sama dengan yang diharapkan Kaggle. Prediksi
# dipetakan berdasarkan id (bukan sekadar urutan baris) agar aman.
pred_by_id = dict(zip(test["id"].values, test_pred))
sub_labels = sample_sub["id"].map(pred_by_id)

# Kalau ada id di sample_submission yang tidak ada di test.csv (tanda dataset
# lokal kadaluarsa), map -> NaN. Isi sementara dengan kelas mayoritas (1)
# hanya agar file valid, TAPI ini bukan solusi -> harus download data terbaru.
n_missing = int(sub_labels.isna().sum())
if n_missing > 0:
    print(f"\n[!!] {n_missing} id di sample_submission tidak ada di test.csv "
          "lokal -> diisi label mayoritas (1) sebagai penambal sementara.")
    print("[!!] Ini TIDAK akan akurat. Download test.csv terbaru dari Kaggle.")
    sub_labels = sub_labels.fillna(1)

submission = pd.DataFrame({
    "id": sample_sub["id"],
    "label": sub_labels.astype(int),
})
submission.to_csv(OUT_PATH, index=False)
print(f"\nSubmission tersimpan di: {OUT_PATH} ({len(submission)} baris)")
print(submission.head())

# ---------------------------------------------------------------------------
# Ide pengembangan lebih lanjut (untuk makalah & iterasi berikutnya):
#   - Ganti/tambah TF-IDF dengan sentence embeddings (mis. IndoBERT,
#     Sentence-Transformers multilingual) untuk menangkap makna semantik,
#     bukan cuma kemiripan kata (pastikan model open-weight / bisa diunduh
#     lokal sesuai aturan lomba -- tidak boleh API tertutup).
#   - Tambah fitur NER sederhana (nama orang/lokasi/tanggal) dgn
#     spaCy/Stanza model Indonesia untuk cek konsistensi tokoh/lokasi.
#   - Coba model lain: SVM, XGBoost/LightGBM di atas fitur yang sama,
#     lalu bandingkan macro F1-nya.
#   - Coba fine-tune IndoBERT sebagai pasangan kalimat (title, content)
#     -> binary classification (butuh GPU, misal Google Colab gratis).
#   - Analisis error: lihat contoh yang salah prediksi, apakah pola
#     tertentu (topik sama tokoh beda, angka beda, dsb).
# ---------------------------------------------------------------------------
