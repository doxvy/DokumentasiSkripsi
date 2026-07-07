# 📦 Studi Komparatif XGBoost dan TabNet untuk Peramalan Permintaan Produk Ritel

> Skripsi — Jurusan Teknik Elektro, Universitas Lampung  
> **Amando Yuviano**  
> Studi Kasus: Toko Plastik Ombo Jaya, Pringsewu, Lampung

---

## 📋 Deskripsi

Penelitian ini membandingkan performa model *machine learning* **XGBoost** dan model *deep learning* **TabNet** dalam melakukan peramalan permintaan (*demand forecasting*) produk ritel pada skala UMKM. Produk yang menjadi objek peramalan adalah produk **Grup A** hasil **Analisis ABC**, yang diklasifikasikan lebih lanjut ke dalam empat kategori pola permintaan menggunakan metode **ADI-CV²** (*Average Demand Interval – Coefficient of Variation Squared*):

| Kategori | Jumlah Produk |
|---|---|
| Smooth | 44 produk |
| Erratic | 42 produk |
| Intermittent | 20 produk |
| Lumpy | 38 produk |

Metodologi penelitian mengacu pada kerangka kerja **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) dan hasil penelitian diintegrasikan ke dalam prototipe aplikasi web berbasis **Streamlit**.

---

## 📁 Struktur Repository

```
📦 repository/
├── 📓 Modeling.ipynb              # Notebook utama: EDA, preprocessing, modeling, evaluasi
│
├── 📂 model/                      # File model hasil training (10 model)
│   ├── xgb_all.pkl
│   ├── xgb_smooth.pkl
│   ├── xgb_erractic.pkl
│   ├── xgb_intermittent.pkl
│   ├── xgb_lumpy.pkl
│   ├── tabnet_all.zip
│   ├── tabnet_smooth.zip
│   ├── tabnet_erractic.zip
│   ├── tabnet_intermittent.zip
│   └── tabnet_lumpy.zip
│
├── 📂 data/                       # File CSV hasil preprocessing dan visualisasi
│   ├── master_konversi.csv        # Tabel konversi satuan produk
│   └── ...                        # File CSV hasil processing lainnya
│
├── 📂 assets/                     # Gambar dan visualisasi hasil analisis
│
├── 🐍 app.py                      # Entry point aplikasi web Streamlit
├── 🐍 processing.py               # Modul pipeline preprocessing dan prediksi
│
├── requirements.txt               # Daftar dependensi Python
└── README.md
```

---

## 🔬 Metodologi

```
Business         Data            Data            Modeling        Evaluation      Deployment
Understanding -> Understanding -> Preparation ->  ----------- -> ----------- -> -----------
                                                  XGBoost         MAE             Streamlit
                                                  TabNet          R² Score        Web App
                                                  (per kategori)  LOR
```

### Pipeline Utama

1. **Analisis ABC** — Klasifikasi 718 produk ke dalam Grup A/B/C berdasarkan kontribusi pendapatan (80/20 split menggunakan `round()`)
2. **Konversi Satuan** — Penyeragaman satuan produk Grup A menggunakan `master_konversi.csv`
3. **Agregasi Mingguan** — Resampling data harian ke mingguan (Senin–Minggu)
4. **Klasifikasi ADI-CV²** — Pengelompokan 144 produk Grup A ke dalam 4 kategori pola permintaan
5. **Feature Engineering** — Pembuatan 24 fitur: lag (1–12 minggu), rolling statistics (avg/std/max/min/median, window 4/8/12 minggu), fitur temporal
6. **Train-Test Split** — Pembagian 80:20 berbasis waktu (batas training: 14 September 2025)
7. **Hyperparameter Tuning** — Optimasi menggunakan **Optuna** (TPE Sampler) untuk kedua model
8. **Evaluasi** — MAE, R² Score, LOR per kategori dan keseluruhan produk Grup A
9. **Deployment** — Integrasi model ke aplikasi web Streamlit

---

## 📊 Hasil Evaluasi

### MAE Test

| Model | All | Smooth | Erratic | Intermittent | Lumpy |
|---|---|---|---|---|---|
| XGBoost | 9,11 | 7,13 | 16,15 | 2,63 | 6,80 |
| TabNet | 9,36 | **6,01** | 16,58 | **2,60** | 7,15 |

### R² Score

| Model | All | Smooth | Erratic | Intermittent | Lumpy |
|---|---|---|---|---|---|
| XGBoost | 0,42 | 0,74 | 0,32 | 0,51 | -0,04 |
| TabNet | 0,45 | **0,88** | 0,30 | **0,57** | -0,05 |

### LOR (Logarithmic Overfitting Ratio)

| Model | All | Smooth | Erratic | Intermittent | Lumpy |
|---|---|---|---|---|---|
| XGBoost | -0,04 | -0,00 | -0,07 | -0,23 | 0,05 |
| TabNet | -0,01 | 0,11 | -0,05 | -0,07 | 0,08 |

> **Catatan:** LOR mendekati nol mengindikasikan model tidak mengalami overfitting maupun underfitting.

---

## ⚙️ Fitur Web Streamlit

| Fitur | Deskripsi |
|---|---|
| 📥 Import Data | Upload file transaksi penjualan dan file konversi satuan |
| 📊 Analisis ABC | Klasifikasi produk dan visualisasi kontribusi pendapatan |
| 🔍 Klasifikasi Demand | Pengelompokan produk berdasarkan pola permintaan ADI-CV² |
| 🔮 Peramalan | Konfigurasi model, kategori produk, dan horizon peramalan 1–16 minggu |
| 📤 Ekspor Hasil | Unduh hasil prediksi dalam format CSV |

---

## 📦 Dependensi Utama

```
xgboost
pytorch-tabnet
optuna
pandas
numpy
scikit-learn
matplotlib
streamlit
```

> Lihat `requirements.txt` untuk daftar lengkap beserta versi yang digunakan.

---

