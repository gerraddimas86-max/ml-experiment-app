# ML Experiment App

Aplikasi antarmuka interaktif untuk eksperimen Machine Learning menggunakan **Streamlit**.
Mendukung **Supervised Learning** (klasifikasi) dan **Unsupervised Learning** (clustering) lewat antarmuka web yang user-friendly.

---

## Fitur

### Supervised Learning (Klasifikasi)
- Upload dataset CSV / Excel
- Pilih kolom target
- Abaikan kolom tertentu (misal `User ID`)
- Atur proporsi data latih / uji (test size)
- Algoritma: **Random Forest**, **SVM**, **KNN**
- Penyesuaian hyperparameter via slider
- Metrik evaluasi: **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrix** (heatmap)
- Preview sampel **data latih** dan **data uji** sebelum dilatih

### Unsupervised Learning (Clustering)
- Upload dataset CSV / Excel
- Pilih kolom fitur (multi-select)
- Algoritma: **K-Means**, **DBSCAN**
- Atur jumlah cluster / `eps` / `min_samples`
- **Silhouette Score** (otomatis `N/A` bila cluster tidak valid)
- **Scatter plot** hasil clustering dalam 2D (reduksi PCA)
- Bar chart distribusi anggota cluster

---

## Struktur Proyek
├── app.py # Entry point (landing page)
├── requirements.txt # Dependensi
├── README.md
├── pages/
│ ├── Unsupervised.py # Halaman clustering
│ └── Supervised.py # Halaman klasifikasi
└── utils/
├── init.py
├── supervised.py # Pipeline klasifikasi
└── unsupervised.py # Pipeline clustering


---

## Cara Menjalankan (Lokal)

### 1. Clone / unduh repositori

```bash
git clone <url-repo-anda> ml-experiment-app
cd ml-experiment-app

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py