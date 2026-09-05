# Panduan Lengkap DAC IFest 2026 (Babak Penyisihan)
### Untuk tim pemula — dibaca dari atas ke bawah

Dokumen ini menjelaskan **datasetnya**, **apa yang harus dikerjakan**,
**langkah demi langkah**, dan **bagaimana menyesuaikan dengan kriteria
penilaian lomba**. Semua angka di sini diambil dari dataset asli yang
sudah dianalisis (bukan tebakan).

---

> ## ⚠️ UPDATE PENTING (dataset terbaru — WAJIB BACA)
> Panitia sudah memperbarui dataset. Setelah dianalisis ulang dengan data baru,
> ada **temuan besar yang mengubah strategi**:
>
> - **Dataset baru JAUH lebih sulit.** Baseline TF-IDF + overlap kata yang tadinya
>   Macro F1 **~0.88**, di dataset baru **anjlok jadi ~0.54** (nyaris sama dengan
>   asal-tebak "Sesuai" yang = 0.47).
> - **Kenapa?** Di data lama, "Tidak Sesuai" = judul & isi beda topik total
>   (overlap kata 0.34 vs 0.80 → gampang dibedakan). Di **data baru**, judul & isi
>   **topiknya sama, kata-katanya mirip** (overlap 0.75 vs 0.80 → hampir sama!).
>   Yang beda adalah **FAKTA-nya**: nama orang, lokasi, jabatan, atau angka di
>   judul **tidak cocok** dengan isi.
>   - Contoh label 0 baru: judul *"Wagub **Jabar** Ancam Denda **Rp100 Juta**..."*
>     tapi isi bilang *"Wakil Gubernur **DKI** Jakarta... denda **dua kali lipat**"*
>     → topik sama (denda pelanggaran prokes), tapi **lokasi & angka beda**.
> - **Konsekuensi strategi:** fitur leksikal (overlap kata/angka/TF-IDF cosine)
>   **tidak lagi cukup**. Untuk menang, arah utama harus ke **pemahaman makna &
>   konsistensi fakta** → **fine-tune IndoBERT sebagai pair classification**
>   (judul vs isi). Lihat Bagian 5 (sudah diperbarui) & Bagian 7.
> - Baseline tetap berguna sebagai **pembanding "lower bound"** di makalah
>   (tunjukkan kenapa pendekatan leksikal gagal → itu justru analisis bernilai).

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
| `train.csv` | 14.397 | `id, title, content, label` | data untuk melatih model (ada jawaban `label`) |
| `test.csv` | 3.603 | `id, title, content` | data untuk diprediksi (TIDAK ada `label`) |
| `sample_submission.csv` | 3.603 | `id, label` | contoh format jawaban yang diunggah ke Kaggle |

> **Catatan:** `id` sekarang berupa teks (`tr…` untuk train, `te00000`–`te03602`
> untuk test), bukan angka. Jumlah `test.csv` = `sample_submission.csv` = 3.603
> (sudah cocok). Script sudah dibuat memetakan prediksi berdasarkan `id`.

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

3. **Fitur leksikal TIDAK lagi memisahkan kelas (ini kunci dataset baru!):**
   - word overlap ratio: label 0 = **0.75** vs label 1 = **0.80** (nyaris sama)
   - overlap angka: label 0 = **0.955** vs label 1 = **0.960** (nyaris sama)
   - overlap kata berkapital (proxy nama/tempat): 0.339 vs 0.385 (lemah)
   - **Artinya:** kalian **tidak bisa** menang hanya dengan menghitung kesamaan
     kata/angka. Judul & isi yang "Tidak Sesuai" pun topik & kosakatanya mirip.
     Perbedaannya ada di **detail fakta** (siapa/di mana/berapa) → butuh model
     yang memahami **makna**, bukan sekadar mencocokkan kata.

4. **Contoh nyata label 0 (Tidak Sesuai) di dataset baru:**
   - Judul: *"Wagub **Jabar** Ancam Denda **Rp100 Juta** Bila Rizieq Melanggar"*
   - Isi: *"Wakil Gubernur **DKI Jakarta** Ahmad Riza Patria... denda **dua kali
     lipat**..."*
   - → Topik sama (sanksi pelanggaran prokes Rizieq), banyak kata sama, tapi
     **lokasi (Jabar vs DKI) & angka (Rp100 juta vs 2x lipat) BEDA**. Inilah
     "misleading headline" yang harus dideteksi model — jauh lebih halus
     daripada dataset versi lama.

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

**Skor baseline di DATASET BARU (validasi hold-out 80/20, seed=42):**
- Macro F1 @ threshold 0.5   : **~0.52**
- Macro F1 @ threshold optimal : **~0.54**
- (bandingkan: asal-tebak "Sesuai" terus = **0.47**)

Jadi di dataset baru, baseline leksikal ini **hampir tidak berguna** — hanya
sedikit di atas tebakan buta. Ini **bukti** (yang bagus untuk ditulis di
makalah) bahwa masalahnya butuh pendekatan semantik, bukan leksikal.
Baseline ini tetap dipakai sebagai **titik pembanding (lower bound)**.

---

## 5. Cara Mengembangkan (STRATEGI DIREVISI untuk dataset baru)

Karena fitur leksikal sudah terbukti lemah (Bagian 1 & 4), **prioritas utama
kalian adalah pendekatan semantik (IndoBERT).** Urutan di bawah sudah diurut
ulang sesuai dampaknya di dataset baru.

**Level 1 — jalankan baseline sebagai pembanding (cepat, wajib):**
- Jalankan `baseline_model.py` → dapat submission valid + skor lower bound (~0.54).
- Ganti validasi hold-out 80/20 → **StratifiedKFold (5 fold)** agar skor stabil.
- Di makalah: tulis bahwa pendekatan leksikal **gagal** & jelaskan kenapa
  (overlap kelas 0 vs 1 hampir sama). Ini analisis bernilai, bukan kegagalan.

**Level 2 — PRIORITAS UTAMA: fine-tune IndoBERT (pair classification):**
- Ini pendekatan yang paling mungkin menang di dataset baru.
- Input: `judul [SEP] isi` → output 0/1. Model belajar apakah isi **mendukung
  fakta** yang diklaim judul (mirip tugas NLI / entailment).
- Model open-weight yang boleh dipakai: **IndoBERT**
  (`indobenchmark/indobert-base-p1`) atau **IndoBERT-large**. Pakai library
  HuggingFace `transformers` + `Trainer`.
- Butuh GPU → **Google Colab gratis** (T4). 2–3 epoch biasanya cukup.
- WAJIB: set `seed`, pakai `class_weight`/weighted loss atau oversampling
  untuk imbalance 90/10, dan **StratifiedKFold** untuk melaporkan skor.
- Karena isi bisa panjang (>300 kata) dan BERT batasnya 512 token, coba juga:
  ambil judul penuh + potongan awal isi (biasanya fakta utama ada di awal).

**Level 3 — fitur "konsistensi fakta" (untuk model klasik / gabungan):**
- Kalau belum bisa BERT, buat fitur yang menandai **ketidakcocokan fakta**:
  - selisih **entitas** (nama orang/lokasi/lembaga) judul vs isi → pakai NER
    Bahasa Indonesia (mis. model `cahya/…` atau spaCy/Stanza) — lebih akurat
    daripada sekadar kata berkapital.
  - selisih **angka & satuan** (Rp, juta, ribu, tanggal) judul vs isi.
  - embedding similarity judul vs isi (IndoBERT/SBERT) sebagai 1 fitur numerik.
- Masukkan fitur ini ke **SVM / XGBoost / LightGBM** (AutoML tetap DILARANG).

**Level 4 — analisis error (nilai "Analisis & Interpretasi" 25%):**
- Lihat contoh yang salah diprediksi. Polanya apa? (entitas beda? angka beda?
  bagian isi yang relevan ada di akhir sehingga terpotong?) Tulis temuan ini di
  makalah — ini yang membedakan tim juara dari tim biasa.

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
