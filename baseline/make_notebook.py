"""Generator notebook IndoBERT pair-classification untuk DAC IFest 2026.

Menghasilkan file .ipynb yang valid & siap dijalankan di Google Colab / Kaggle.
Jalankan: python3 baseline/make_notebook.py
"""
import json
import os

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------------------
md(r"""# Penyisihan DAC IFest 2026 — Deteksi Kesesuaian Judul & Isi Berita
## Model utama: **IndoBERT (fine-tuning sebagai pair-classification)**

**Nama Tim:** `<ISI_NAMA_TIM>`  ·  **Anggota:** `<ANGGOTA 1, 2, 3>`

---

### Deklarasi (WAJIB sesuai guidebook)
- **Pre-trained model (open-weight):** `indobenchmark/indobert-base-p1`
  (IndoBERT, publik & dapat diunduh penuh dari HuggingFace) —
  https://huggingface.co/indobenchmark/indobert-base-p1
- **Dataset eksternal:** _tidak ada_ (hanya memakai dataset resmi panitia).
  > Jika nanti memakai dataset eksternal, cantumkan link Google Drive-nya di sini.
- **Seed** untuk semua randomness: `SEED = 42` (lihat sel Konfigurasi).
- **AutoML tidak digunakan** — arsitektur & hyperparameter dipilih manual.

### Ringkasan pendekatan
Tugasnya: menebak apakah **judul (title)** sesuai dengan **isi (content)**
berita (label 1 = Sesuai, 0 = Tidak Sesuai). Ini masalah **klasifikasi
pasangan teks**. Dari EDA, fitur leksikal (overlap kata/angka) **tidak** memisah
kelas dengan baik pada dataset ini — perbedaan "Tidak Sesuai" bersifat **fakta**
(nama/lokasi/angka berbeda) meski topik & kata mirip. Karena itu kami memakai
**IndoBERT** yang memahami makna & konteks, di-fine-tune sebagai pasangan
`(judul, isi) -> 0/1`.

Teknik penting yang dipakai:
1. **Weighted loss** untuk menangani imbalance 90/10.
2. **Truncation `only_second`**: judul dipertahankan penuh, isi dipotong
   (fakta utama berita umumnya di awal).
3. **Stratified K-Fold ensemble** + **threshold tuning** untuk Macro F1.
""")

# ---------------------------------------------------------------------------
md(r"""## 1. Setup — install & import
Jalankan di **Google Colab** (Runtime → Change runtime type → **GPU T4**) atau
**Kaggle Notebook** (Settings → Accelerator → GPU).""")

code(r"""
# Colab biasanya sudah ada torch. Kita pasang/upgrade transformers & tools.
!pip -q install -U "transformers>=4.40" "accelerate>=0.30" scikit-learn pandas numpy
""")

code(r"""
import os, random, glob, zipfile
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed,
)

print("torch:", torch.__version__, "| GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
""")

# ---------------------------------------------------------------------------
md(r"""## 2. Konfigurasi & Seed
`SEED` dikunci agar hasil bisa direproduksi (syarat lomba).

- **FAST_MODE = True** → latih 1 model (split 85/15). Paling cepat (~15–25 menit),
  cocok kalau **ingin submit hari ini**.
- **FAST_MODE = False** → 5-Fold ensemble (lebih akurat & tahan uji di private
  leaderboard, tapi ~5x lebih lama). Pakai ini untuk submission final.""")

code(r"""
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# ---- Konfigurasi utama ----
MODEL_NAME = "indobenchmark/indobert-base-p1"  # open-weight (boleh dipakai)
MAX_LEN    = 256          # panjang token maksimum (judul + potongan isi)
EPOCHS     = 3
LR         = 2e-5
TRAIN_BS   = 16
EVAL_BS    = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

FAST_MODE  = True         # True = 1 model cepat; False = 5-fold ensemble
N_FOLDS    = 5            # dipakai saat FAST_MODE = False
OUT_PATH   = "submission.csv"

use_fp16 = torch.cuda.is_available()
print("FAST_MODE:", FAST_MODE, "| fp16:", use_fp16)
""")

# ---------------------------------------------------------------------------
md(r"""## 3. Load data (otomatis untuk Colab & Kaggle)
- **Kaggle Notebook:** data otomatis ada di `/kaggle/input/...` — langsung jalan.
- **Google Colab:** kalau file belum ada, sel di bawah akan meminta kamu
  **upload** zip dataset (mis. `penyisihan-ifest-2026-dac.zip`) lalu meng-extract.""")

code(r"""
def find_csv(name):
    pats = [f"/kaggle/input/**/{name}", f"**/{name}"]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None

# Kalau train.csv belum ketemu (khas Colab), minta upload zip lalu extract.
if find_csv("train.csv") is None:
    try:
        from google.colab import files
        print("Silakan upload zip dataset (mis. penyisihan-ifest-2026-dac.zip)...")
        up = files.upload()
        for fn in up:
            if fn.endswith(".zip"):
                with zipfile.ZipFile(fn) as z:
                    z.extractall("data")
                print("Extracted:", fn)
    except Exception as e:
        print("Bukan di Colab / upload dilewati:", e)

train_path  = find_csv("train.csv")
test_path   = find_csv("test.csv")
sample_path = find_csv("sample_submission.csv")
assert train_path and test_path and sample_path, \
    "CSV tidak ditemukan. Pastikan dataset ter-upload/ter-mount."
print("Memakai:\n ", train_path, "\n ", test_path, "\n ", sample_path)

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)
print("train:", train.shape, "| test:", test.shape, "| sample:", sample_sub.shape)
train.head(3)
""")

# ---------------------------------------------------------------------------
md(r"""## 4. EDA singkat & pengecekan
Cek distribusi label (imbalance) dan pastikan tidak ada teks kosong.""")

code(r"""
print("Missing train:", train[["title","content","label"]].isna().sum().to_dict())
print("Missing test :", test[["title","content"]].isna().sum().to_dict())
print("\nDistribusi label (train):")
print(train["label"].value_counts())
print((train["label"].value_counts(normalize=True)*100).round(2).astype(str)+" %")

# amankan teks kosong -> string kosong
for df in (train, test):
    df["title"]   = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
""")

# ---------------------------------------------------------------------------
md(r"""## 5. Tokenisasi (pair) & Dataset
Judul dan isi dimasukkan sebagai **pasangan kalimat**. Dengan
`truncation="only_second"`, **judul dipertahankan penuh** dan hanya **isi** yang
dipotong bila melebihi `MAX_LEN` (fakta utama berita umumnya di bagian awal).""")

code(r"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(titles, contents):
    return tokenizer(
        list(titles), list(contents),
        truncation="only_second", max_length=MAX_LEN, padding=False,
    )

class NewsPairDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels=None):
        self.enc = enc
        self.labels = labels
    def __len__(self):
        return len(self.enc["input_ids"])
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[i]))
        return item

test_enc = encode(test["title"], test["content"])
test_ds  = NewsPairDataset(test_enc)
""")

# ---------------------------------------------------------------------------
md(r"""## 6. Model, metrik, & Trainer dengan weighted loss
Karena data imbalance 90/10, kami memberi **bobot lebih besar pada kelas
minoritas (0)** di fungsi loss agar model tidak malas menebak "Sesuai" terus.""")

code(r"""
from collections import Counter
cnt = Counter(train["label"].tolist())
n_total = sum(cnt.values())
# bobot inversely proportional terhadap frekuensi kelas
class_weights = torch.tensor(
    [n_total / (2.0 * cnt[c]) for c in [0, 1]], dtype=torch.float
)
print("class_weights [kelas0, kelas1]:", class_weights.tolist())

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(outputs.logits.device)
        )
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def make_trainer(model, train_ds, val_ds, fold_tag):
    args = TrainingArguments(
        output_dir=f"out_{fold_tag}",
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=100,
        report_to="none",
        seed=SEED,
        save_total_limit=1,
    )
    return WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
    )

def softmax_prob1(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p[:, 1]
""")

# ---------------------------------------------------------------------------
md(r"""## 7. Training (K-Fold ensemble atau single split)
Untuk tiap fold: latih IndoBERT, simpan **probabilitas validasi (OOF)** untuk
threshold tuning, dan **akumulasi probabilitas test** (dirata-rata di akhir =
ensemble). Ini memberi skor lebih stabil & tahan uji di private leaderboard.""")

code(r"""
X_title = train["title"].values
X_content = train["content"].values
y = train["label"].values

oof_prob = np.zeros(len(train))     # prob kelas 1 utk data train (out-of-fold)
oof_mask = np.zeros(len(train), dtype=bool)
test_prob = np.zeros(len(test))     # akumulasi prob test dari tiap fold
n_models = 0

if FAST_MODE:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    splits = list(splitter.split(X_title, y))
else:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_title, y))

for fold, (tr_idx, va_idx) in enumerate(splits):
    print(f"\n{'='*60}\nFOLD {fold+1}/{len(splits)}  "
          f"(train={len(tr_idx)}, val={len(va_idx)})\n{'='*60}")

    tr_enc = encode(X_title[tr_idx], X_content[tr_idx])
    va_enc = encode(X_title[va_idx], X_content[va_idx])
    tr_ds = NewsPairDataset(tr_enc, y[tr_idx])
    va_ds = NewsPairDataset(va_enc, y[va_idx])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    trainer = make_trainer(model, tr_ds, va_ds, fold_tag=fold)
    trainer.train()

    # prob validasi (OOF)
    va_logits = trainer.predict(va_ds).predictions
    oof_prob[va_idx] = softmax_prob1(va_logits)
    oof_mask[va_idx] = True

    # prob test (akumulasi utk ensemble)
    te_logits = trainer.predict(test_ds).predictions
    test_prob += softmax_prob1(te_logits)
    n_models += 1

    del model, trainer
    torch.cuda.empty_cache()

test_prob /= n_models
print(f"\nSelesai training {n_models} model.")
""")

# ---------------------------------------------------------------------------
md(r"""## 8. Threshold tuning untuk Macro F1
Karena imbalance, ambang 0.5 belum tentu optimal. Kami cari threshold yang
memaksimalkan **Macro F1** pada prediksi out-of-fold (data validasi).""")

code(r"""
y_oof = y[oof_mask]
p_oof = oof_prob[oof_mask]

best_th, best_f1 = 0.5, 0.0
for th in np.arange(0.05, 0.96, 0.01):
    f1 = f1_score(y_oof, (p_oof >= th).astype(int), average="macro")
    if f1 > best_f1:
        best_f1, best_th = f1, th

print(f"Macro F1 @0.50           : {f1_score(y_oof,(p_oof>=0.5).astype(int),average='macro'):.4f}")
print(f"Threshold optimal        : {best_th:.2f}")
print(f"Macro F1 @threshold opt  : {best_f1:.4f}")

print("\nClassification report (OOF, threshold optimal):")
print(classification_report(
    y_oof, (p_oof >= best_th).astype(int),
    target_names=["Tidak Sesuai (0)", "Sesuai (1)"]))

print("Confusion matrix (baris=aktual, kolom=prediksi):")
print(confusion_matrix(y_oof, (p_oof >= best_th).astype(int)))
""")

# ---------------------------------------------------------------------------
md(r"""## 9. Buat submission
Prediksi test memakai threshold optimal, lalu output diselaraskan dengan
`sample_submission.csv` (dipetakan berdasarkan `id` supaya jumlah & urutan baris
**persis** seperti yang diharapkan Kaggle).""")

code(r"""
test_pred = (test_prob >= best_th).astype(int)
pred_by_id = dict(zip(test["id"].values, test_pred))

sub_labels = sample_sub["id"].map(pred_by_id)
missing = int(sub_labels.isna().sum())
if missing:
    print(f"[!!] {missing} id sample_submission tak ada di test -> isi 1 sementara.")
    sub_labels = sub_labels.fillna(1)

submission = pd.DataFrame({"id": sample_sub["id"], "label": sub_labels.astype(int)})
submission.to_csv(OUT_PATH, index=False)
print("Tersimpan:", OUT_PATH, "| baris:", len(submission))
print("Distribusi prediksi:", submission["label"].value_counts().to_dict())
submission.head()
""")

code(r"""
# (Colab) unduh submission ke komputer:
try:
    from google.colab import files
    files.download(OUT_PATH)
except Exception as e:
    print("Lewati download otomatis:", e)
""")

# ---------------------------------------------------------------------------
md(r"""## 10. Catatan untuk makalah & peningkatan
**Untuk makalah (nilai besar di penyisihan):**
- Laporkan **Macro F1 (OOF)** di atas — inilah estimasi skor yang jujur.
- Bandingkan dengan baseline leksikal (TF-IDF + LogReg ~0.54) untuk menunjukkan
  **kenapa pendekatan semantik (IndoBERT) diperlukan**.
- Sertakan **confusion matrix** & **analisis error** (contoh yang salah:
  entitas/angka mirip? bagian relevan isi terpotong?).

**Ide peningkatan skor (kalau masih ada waktu):**
1. Set `FAST_MODE = False` → 5-fold ensemble (biasanya naik & lebih stabil).
2. Coba `indobenchmark/indobert-large-p1` (lebih kuat, lebih lambat).
3. Naikkan `MAX_LEN` ke 384/512 bila isi penting terpotong (lebih lambat).
4. Ensembling dengan model lain (mis. `cahya/roberta-base-indonesian` atau
   XLM-R) lalu rata-ratakan probabilitasnya.
5. Tambah fitur "konsistensi fakta" (selisih entitas/angka judul vs isi) sebagai
   model kedua, lalu blend.

**Ingat aturan:** seed sudah di-set, tanpa AutoML, IndoBERT open-weight sudah
dideklarasikan di markdown teratas. Ganti nama file menjadi
`Penyisihan_DAC2026_NamaTim.ipynb` sebelum dikumpulkan.
""")

# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "Penyisihan_DAC2026_NamaTim.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook ditulis:", out, "| jumlah cell:", len(cells))
