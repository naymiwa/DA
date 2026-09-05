# Panduan Lengkap DAC IFest 2026 (Babak Penyisihan)
### Untuk tim pemula — dibaca dari atas ke bawah

Dokumen ini menjelaskan **datasetnya**, **apa yang harus dikerjakan**,
**langkah demi langkah**, dan **bagaimana menyesuaikan dengan kriteria
penilaian lomba**. Semua angka di sini diambil dari dataset asli yang
sudah dianalisis (bukan tebakan).

---

## 0. Ringkasan super singkat (baca ini dulu)

- **Jenis lomba:** klasifikasi teks. Tugasnya: menebak apakah **JUDUL berita
  SESUAI dengan ISI berita** atau tidak.
- **Ini BUKAN persis seperti notebook BBC.** Notebook BBC = menebak *kategori*
  berita (olahraga/politik/dll) dari 1 teks. Lomba ini = membandingkan **2 teks
  (judul vs isi)** dan menjawab **Sesuai / Tidak Sesuai** (2 kelas saja).
  Teknik dasarnya (TF-IDF + model sklearn) mirip, tapi *cara berpikirnya beda*.
  Penjelasan lengkap di Bagian 7.
- **Sudah ada baseline jadi** di folder `baseline/` yang skornya **Macro F1 ~0.88**
  di validasi. Ini titik awal kalian, tinggal dikembangkan.
- **Deadline mepet:** dataset rilis 1 Sept, pengumpulan makalah + Kaggle
  **1–13 September 2026**. Jadi waktu efektif tinggal seminggu-an. Bagian 8
  berisi rencana harian.

---

## 1. Membaca Dataset (hasil analisis nyata)

Dataset ada 3 file (di dalam `penyisihan-dac-ifest-2026 (2).zip`):

| File | Baris | Kolom | Fungsi |
|------|-------|-------|--------|
| `train.csv` | 14.400 | `id, title, content, label` | data untuk melatih model (ada jawaban `label`) |
| `test.csv` | 3.600 | `id, title, content` | data untuk diprediksi (TIDAK ada `label`) |
| `sample_submission.csv` | 3.600 | `id, label` | contoh format jawaban yang diunggah ke Kaggle |

**Arti kolom:**
- `title` = judul berita (rata-rata ~10 kata).
- `content` = isi/badan berita (rata-rata ~275–308 kata).
- `label` = **1 = Sesuai** (judul cocok dengan isi), **0 = Tidak Sesuai**
  (judul tidak nyambung dengan isi).

**Fakta penting dari dataset (WAJIB masuk ke makalah bagian EDA):**

1. **Tidak ada missing value** sama sekali di train maupun test. (Bagus, jadi
   tidak perlu isi/menghapus data kosong — tapi tetap tulis di makalah bahwa
   kalian sudah mengeceknya.)

2. **Data sangat tidak seimbang (imbalance):**
   - label 1 (Sesuai): 12.960 baris = **90%**
   - label 0 (Tidak Sesuai): 1.440 baris = **10%**
   - **Kenapa ini penting?** Kalau model asal menebak "Sesuai" terus, akurasinya
     tetap 90% tapi *jelek* untuk kelas minoritas. Karena itu metriknya pakai
     **Macro F1** (rata-rata F1 dua kelas), bukan akurasi. Kalian **harus**
     menangani imbalance (misalnya `class_weight='balanced'`, atau
     oversampling, atau threshold tuning). Ini poin nilai di kriteria EDA &
     Prapemrosesan (25%).

3. **Sinyal paling kuat = seberapa banyak kata di judul muncul di isi**
   ("word overlap ratio"):
   - label 0 (Tidak Sesuai): rata-rata overlap **0.34**
   - label 1 (Sesuai): rata-rata overlap **0.80**
   - Artinya: kalau banyak kata judul muncul di isi → cenderung "Sesuai".
     Ini fitur emas, tapi **hati-hati**: kadang topik sama tapi peristiwanya
     beda (angka/tanggal/tokoh beda) → tetap "Tidak Sesuai". Itulah tantangan
     utama lomba ini.

4. **Contoh nyata label 0 (Tidak Sesuai):**
   - Judul: *"IDI Sebut Ruang ICU Khusus Covid-19 di Surabaya Sudah Penuh"*
   - Isi: *"Pemerintah Afrika Selatan menunda dimulainya tahun ajaran baru..."*
   - → Judul ngomong ICU Surabaya, isinya soal sekolah di Afrika Selatan.
     Jelas tidak nyambung. Model harus bisa menangkap perbedaan **topik/entitas**
     seperti ini.

> **Catatan metrik:** baseline mengasumsikan metriknya **Macro F1**. Sebelum
> mulai, buka halaman Kaggle "Overview → Evaluation" dan **pastikan metrik
> resminya**. Kalau ternyata beda (misal F1 biasa atau ROC-AUC), sesuaikan
> bagian "threshold tuning" & pelaporan skor. Semua langkah lain tetap sama.

---

## 2. Memahami Tugasnya (framing yang benar)

Ini adalah **binary text-pair classification** (klasifikasi pasangan teks jadi
2 kelas). Dalam dunia nyata masalah ini disebut deteksi
**"headline–content mismatch"** atau semacam **clickbait/misleading headline
detection**. Framing ini berguna untuk bagian **Pendahuluan** & **Kajian
Pustaka** makalah kalian (kenapa masalah ini penting: melawan hoaks/clickbait,
menjaga kualitas jurnalisme, dsb).

Input: (judul, isi) → Output: 0 atau 1.

---

## 3. Alur Kerja Standar (pipeline) — versi mudah dipahami

Bayangkan seperti resep masakan, 7 tahap:

1. **Load data** → baca `train.csv` & `test.csv` pakai pandas.
2. **EDA (Exploratory Data Analysis)** → lihat ukuran, distribusi label,
   panjang teks, contoh per kelas, korelasi fitur. (Sudah disiapkan di
   `baseline/eda.py`.)
3. **Preprocessing (prapemrosesan teks)** → bersihkan teks: huruf kecil,
   buang tanda baca berlebih, (opsional) stopword removing/stemming Bahasa
   Indonesia (pakai library **Sastrawi**).
4. **Feature engineering** → ubah teks jadi angka yang bisa dibaca model:
   - Fitur "hubungan judul-isi": word overlap, angka yang sama, cosine
     similarity TF-IDF judul vs isi, overlap nama berkapital (proxy nama
     orang/tempat), dll.
   - Fitur TF-IDF dari gabungan "judul [SEP] isi".
5. **Modeling** → latih model (Logistic Regression / SVM / XGBoost / IndoBERT).
6. **Validasi & tuning** → ukur Macro F1 pakai **Stratified K-Fold** (bukan
   1 split saja), tuning threshold & hyperparameter.
7. **Prediksi test → buat submission.csv** (kolom `id,label`) → unggah ke Kaggle.

Baseline yang sudah jadi (`baseline/baseline_model.py`) menjalankan tahap 1–7
dengan Logistic Regression + TF-IDF + fitur handcrafted, dan menghasilkan
`baseline/baseline_submission.csv`.

---

## 4. Apa yang Sudah Ada (baseline) & cara menjalankannya

Folder `baseline/`:
- `eda.py` — cetak statistik dataset (Bagian 1 di atas datang dari sini).
- `baseline_model.py` — model baseline lengkap sampai bikin submission.
- `baseline_submission.csv` — hasil prediksi, **bisa langsung diunggah ke
  Kaggle sebagai submission pertama** untuk memastikan format benar.

**Cara menjalankan** (di komputer/Colab, butuh pandas numpy scipy scikit-learn):
```bash
python3 baseline/eda.py
python3 baseline/baseline_model.py
```

**Skor baseline (validasi hold-out 80/20, seed=42):**
- Macro F1 @ threshold 0.5   : **~0.855**
- Macro F1 @ threshold optimal (0.18) : **~0.883**

Threshold 0.18 (bukan 0.5) dipakai karena data imbalance — kita perlu lebih
"peka" mendeteksi kelas minoritas (0).

---

## 5. Cara Mengembangkan (dari baseline → skor lebih tinggi)

Kerjakan berurutan, dari yang paling mudah & berdampak:

**Level 1 — wajib & mudah (rapikan baseline):**
- Ganti validasi hold-out 80/20 → **StratifiedKFold (5 fold)** supaya skor
  yang dilaporkan stabil. (Sudah di-import di `baseline_model.py`, tinggal
  dipakai.) Ini juga nilai plus di kriteria "Metodologi".
- Tambah preprocessing Bahasa Indonesia: **Sastrawi** (stopword removal +
  stemming). Uji apakah menaikkan Macro F1.
- Coba fitur TF-IDF **char n-gram** (menangkap typo & imbuhan) selain word.

**Level 2 — coba model lain (bandingkan, jangan cuma 1):**
- **SVM (LinearSVC)**, **XGBoost**, atau **LightGBM** di atas fitur yang sama.
- Buat tabel perbandingan Macro F1 antar model di makalah (juri suka ini).
- Catatan: **AutoML DILARANG**. Jadi pilih & tuning model manual.

**Level 3 — semantic (paling ampuh untuk teks, butuh sedikit effort):**
- Pakai **sentence embeddings** untuk mengukur kemiripan MAKNA judul vs isi
  (bukan sekadar kata sama). Contoh model open-weight (boleh dipakai sesuai
  aturan): **IndoBERT** (`indobenchmark/indobert-base-p1`) atau
  Sentence-Transformers multilingual. Hitung cosine similarity embedding
  judul vs isi → jadikan fitur tambahan.
- Level tertinggi: **fine-tune IndoBERT** sebagai *pair classification*
  (input: judul + [SEP] + isi → 0/1). Butuh GPU → pakai **Google Colab gratis**.
  Ini biasanya memberi skor terbaik.

**Level 4 — analisis error (nilai "Analisis & Interpretasi" 25%):**
- Lihat contoh yang salah diprediksi. Polanya apa? (topik sama tokoh beda?
  angka beda? judul terlalu pendek?) Tulis temuan ini di makalah — ini yang
  membedakan tim juara dari tim biasa.

> **Aturan yang WAJIB dipatuhi (dari guidebook):**
> - **Seed wajib** di-set untuk semua randomness (sudah: `SEED=42`).
> - **AutoML dilarang.**
> - **Pre-trained model boleh** asal *open-weight* (bisa diunduh lokal, mis.
>   IndoBERT). API tertutup (ChatGPT API dll) tidak untuk jadi model inti.
> - **Dataset eksternal boleh** asal publik, gratis, bukan ground-truth, dan
>   **dideklarasikan di markdown paling atas notebook** + link Google Drive.

---

## 6. Menyesuaikan dengan Kriteria Penilaian (INI KUNCI MENANG)

Nilai penyisihan = **Kaggle Score 30% + Makalah & Notebook 70%**.
Jadi **makalah lebih besar bobotnya daripada skor Kaggle**. Jangan cuma
kejar leaderboard — tulis makalah yang bagus!

Rincian **Makalah & Notebook (70%)** dan cara memenuhinya:

| Aspek | Bobot | Yang harus ada di makalah/notebook |
|-------|-------|-------------------------------------|
| **EDA & Prapemrosesan** | 25% | Statistik dataset, cek missing value, **penanganan imbalance**, feature engineering, **visualisasi** (grafik distribusi label, histogram panjang teks, boxplot word-overlap per kelas, wordcloud). |
| **Metodologi & Pemodelan** | 30% | Alur kerja terstruktur, alasan pemilihan algoritma, teknik optimasi (K-Fold, tuning, threshold), **kode notebook rapi & sinkron dengan makalah**. |
| **Analisis & Interpretasi** | 25% | Kenapa model X dipilih, cara kerjanya, interpretasi metrik (Macro F1, precision/recall per kelas, confusion matrix), **analisis error**. |
| **Rekomendasi Strategis & Kualitas Makalah** | 20% | Terjemahkan hasil teknis → rekomendasi nyata (mis. sistem deteksi judul menyesatkan untuk media/platform), tata bahasa rapi (PUEBI), struktur IEEE. |

**Aturan format makalah (guidebook):**
- Format **IEEE** (template: ieee.org/conferences/publishing/templates.html).
- **Bahasa Indonesia** sesuai **PUEBI**.
- **Maksimal 10 halaman.**
- Minimal 3 bagian: **Pendahuluan, Kajian Pustaka, Metodologi**
  (tambahkan juga: Hasil & Pembahasan, Kesimpulan, Rekomendasi).
- Nama file:
  - Notebook: `Penyisihan_DAC2026_NamaTim.ipynb` (wajib ada **seed**).
  - Makalah: `Makalah_DAC2026_NamaTim.pdf`.
- Dikumpulkan lewat **website resmi IFest** oleh perwakilan tim.

**Kerangka makalah yang disarankan (petakan ke bobot nilai di atas):**
1. **Abstrak** — ringkasan masalah, metode, hasil (Macro F1), 1 paragraf.
2. **Pendahuluan** — latar belakang (bahaya judul menyesatkan/clickbait),
   manfaat, tujuan analisis.
3. **Kajian Pustaka** — teori klasifikasi teks, TF-IDF, (IndoBERT/embedding),
   penanganan imbalance, referensi penelitian sejenis.
4. **Metodologi** — dataset, EDA, preprocessing, feature engineering, model,
   validasi (K-Fold), metrik. → penuhi bobot 25% + 30%.
5. **Hasil & Pembahasan** — tabel perbandingan model, confusion matrix,
   interpretasi, analisis error. → penuhi bobot 25%.
6. **Rekomendasi Strategis** — solusi aplikatif dari model. → penuhi bobot 20%.
7. **Kesimpulan** & **Daftar Pustaka**.

---

## 7. Apakah mirip notebook BBC? (jawaban jujur)

**Mirip sebagian, tapi jangan ditiru mentah-mentah.**

| | Notebook BBC | Lomba DAC ini |
|--|--------------|----------------|
| Jenis | Multi-class (5 kategori topik) | **Binary** (Sesuai / Tidak Sesuai) |
| Input | **1 teks** (artikel) | **2 teks** (judul + isi) yang harus dibandingkan |
| Inti masalah | "Ini berita tentang apa?" | "Apakah judul cocok dengan isi?" |
| Data | biasanya seimbang | **sangat imbalance (90/10)** → butuh penanganan khusus |
| Metrik | akurasi | **Macro F1** |

**Yang bisa dicontoh dari notebook BBC:** alur umum (load → bersihkan teks →
TF-IDF → model sklearn → evaluasi). Bagus untuk **belajar dasar**.

**Yang HARUS beda di lomba ini:** kalian tidak cukup meng-TF-IDF satu teks.
Kekuatan solusi ada di **fitur yang membandingkan judul vs isi** (overlap kata,
kesamaan angka/tanggal, cosine similarity, kesamaan entitas) dan **penanganan
imbalance**. Baseline di repo ini sudah melakukannya — jadikan itu fondasi.

---

## 8. Rencana Kerja untuk 3 Orang (deadline 13 Sept)

Bagi tugas biar paralel. Semua tetap review bareng.

- **Orang A — Data & EDA:** jalankan `eda.py`, buat semua **visualisasi**
  (distribusi label, histogram panjang, boxplot overlap per kelas, wordcloud),
  tulis bagian **EDA & Prapemrosesan** makalah.
- **Orang B — Modeling:** kembangkan `baseline_model.py` (K-Fold, coba SVM/
  XGBoost, lalu embedding/IndoBERT), buat tabel perbandingan, kelola submission
  Kaggle (ingat **batas submission per hari**), tulis bagian **Metodologi &
  Hasil**.
- **Orang C — Makalah & Analisis:** susun makalah IEEE, **Pendahuluan +
  Kajian Pustaka + Rekomendasi**, analisis error bareng Orang B, rapikan PUEBI,
  cek maksimal 10 halaman.

**Timeline saran:**
- **Hari 1–2:** semua paham dataset (baca dokumen ini), jalankan baseline,
  submit `baseline_submission.csv` ke Kaggle untuk cek format & dapat skor awal.
- **Hari 3–4:** EDA + visualisasi selesai; modeling Level 1–2 (K-Fold, SVM/XGB).
- **Hari 5–6:** coba Level 3 (embedding/IndoBERT) kalau sempat; analisis error.
- **Hari 7:** finalisasi model & submission terbaik.
- **Hari 8+:** finalisasi makalah, proofread PUEBI, cek nama file, kumpulkan.

---

## 9. Checklist sebelum submit (jangan sampai gugur!)

- [ ] Submission Kaggle formatnya `id,label` (persis `sample_submission.csv`).
- [ ] Notebook bernama `Penyisihan_DAC2026_NamaTim.ipynb` dan **ada seed**.
- [ ] Makalah PDF bernama `Makalah_DAC2026_NamaTim.pdf`, format IEEE, ≤10 hal,
      Bahasa Indonesia PUEBI.
- [ ] Tidak pakai AutoML.
- [ ] Kalau pakai dataset eksternal / pretrained: dideklarasikan di markdown
      teratas notebook + link Google Drive yang bisa diakses.
- [ ] Kalau pakai pretrained model: pastikan **open-weight**.
- [ ] Notebook & makalah **konsisten** (angka/metode yang ditulis = yang di kode).
- [ ] Dikumpulkan lewat website resmi IFest sebelum **13 September 2026**.

---

Semangat! Baseline sudah kuat (Macro F1 ~0.88). Fokus terbesar untuk menang:
**makalah yang rapi + analisis mendalam (70% nilai)**, bukan cuma skor Kaggle.
