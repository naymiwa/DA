"""
EDA cepat untuk kompetisi DAC IFest 2026 - Kesesuaian Judul & Isi Berita.

Cara pakai:
    python3 baseline/eda.py

Akan mencetak ringkasan dataset ke terminal (dan bisa disalin ke bagian
"Eksplorasi Data Analisis (EDA)" di makalah).
"""

import os
import zipfile
import glob
import pandas as pd
import numpy as np
import re

DATA_DIR = "data/penyisihan-dac-ifest-2026"

if not os.path.exists(f"{DATA_DIR}/train.csv"):
    # ekstrak otomatis dari zip Kaggle yang ada di root repo, kalau
    # folder data/ belum ada (misal setelah clone ulang / git ignore data/)
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

print("=" * 60)
print("UKURAN DATASET")
print("=" * 60)
print(f"train.csv : {train.shape[0]} baris, {train.shape[1]} kolom -> {list(train.columns)}")
print(f"test.csv  : {test.shape[0]} baris, {test.shape[1]} kolom -> {list(test.columns)}")

print()
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print("train:\n", train.isnull().sum())
print("test:\n", test.isnull().sum())

print()
print("=" * 60)
print("DISTRIBUSI LABEL (train) -- PENTING: cek imbalance!")
print("=" * 60)
counts = train["label"].value_counts().sort_index()
pct = train["label"].value_counts(normalize=True).sort_index() * 100
for lbl in counts.index:
    print(f"label={lbl} ({'Tidak Sesuai' if lbl == 0 else 'Sesuai'}): "
          f"{counts[lbl]} baris ({pct[lbl]:.2f}%)")
print()
print(">> Dataset SANGAT imbalance (sekitar 90% Sesuai vs 10% Tidak Sesuai).")
print(">> Karena metrik lomba Macro F1, kelas minoritas (0) harus tetap")
print(">> diprediksi dengan baik -- jangan sampai model hanya menebak label 1 terus.")

print()
print("=" * 60)
print("PANJANG TEKS (jumlah kata)")
print("=" * 60)
train["title_len"] = train["title"].astype(str).str.split().apply(len)
train["content_len"] = train["content"].astype(str).str.split().apply(len)
print(train[["title_len", "content_len"]].describe())

print()
print("Panjang judul & isi per kelas label (median):")
print(train.groupby("label")[["title_len", "content_len"]].median())

print()
print("=" * 60)
print("CONTOH DATA")
print("=" * 60)
for lbl in sorted(train["label"].unique()):
    row = train[train["label"] == lbl].iloc[0]
    tag = "Sesuai" if lbl == 1 else "Tidak Sesuai"
    print(f"\n--- Contoh label={lbl} ({tag}) ---")
    print("Judul :", row["title"])
    print("Isi   :", row["content"][:300], "...")

print()
print("=" * 60)
print("OVERLAP KATA SEDERHANA antara judul & isi (word overlap ratio)")
print("=" * 60)


def word_overlap(title, content):
    t_words = set(re.findall(r"\w+", str(title).lower()))
    c_words = set(re.findall(r"\w+", str(content).lower()))
    if not t_words:
        return 0.0
    return len(t_words & c_words) / len(t_words)


train["overlap_ratio"] = [
    word_overlap(t, c) for t, c in zip(train["title"], train["content"])
]
print(train.groupby("label")["overlap_ratio"].describe())
print()
print(">> Bandingkan rata-rata overlap_ratio antar label. Kalau berbeda jauh,")
print(">> fitur overlap kata ini sudah cukup informatif untuk model baseline.")
print(">> Tapi ingat: overlap tinggi belum tentu 'Sesuai' (bisa topik sama,")
print(">> peristiwa beda) -- itulah tantangan utama lomba ini.")
