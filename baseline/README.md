# Baseline - DAC IFest 2026 (Kesesuaian Judul & Isi Berita)

Folder ini berisi starter code untuk kompetisi Data Analysis Competition
IFest 2026 (Kaggle: `penyisihan-dac-ifest-2026`).

## Isi

- `eda.py` — eksplorasi data cepat (ukuran data, distribusi label, panjang
  teks, overlap kata judul-isi, contoh data per kelas).
- `baseline_model.py` — model baseline: fitur handcrafted (overlap kata,
  overlap angka, cosine similarity TF-IDF judul-isi, dll) + TF-IDF +
  Logistic Regression (`class_weight="balanced"` karena label imbalance
  90%/10%), dengan threshold tuning untuk Macro F1. Skrip ini
  langsung menghasilkan `baseline_submission.csv` yang formatnya sudah
  sesuai `sample_submission.csv` (kolom `id,label`).
- `baseline_submission.csv` — hasil prediksi baseline pada test set (bisa
  langsung diunggah ke Kaggle sebagai submission pertama).

## Cara menjalankan

```bash
python3 baseline/eda.py
python3 baseline/baseline_model.py
```

Membutuhkan: pandas, numpy, scipy, scikit-learn.

## Hasil baseline (hold-out validation 80/20, seed=42)

- Macro F1 @ threshold 0.5   : ~0.855
- Macro F1 @ threshold optimal (0.18) : ~0.883

Ini baseline sederhana berbasis kemiripan leksikal (TF-IDF, overlap kata,
overlap angka). Skor ini adalah **titik awal**, bukan hasil akhir — lihat
bagian "Ide pengembangan lebih lanjut" di akhir `baseline_model.py` dan
panduan lengkap dari asisten untuk langkah peningkatan (semantic embedding,
fitur NER, model lain, dsb).

**Catatan penting:** sebelum submit ke Kaggle & ditulis di makalah, ganti
validasi hold-out 80/20 di `baseline_model.py` dengan `StratifiedKFold`
(sudah di-import di skrip) agar skor Macro F1 yang dilaporkan lebih stabil
dan tidak bergantung pada satu split acak saja.
