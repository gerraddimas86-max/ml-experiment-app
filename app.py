import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Experiment App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- HEADER ----------------
st.title("🤖 ML Experiment App")
st.caption(
    "Antarmuka interaktif untuk eksperimen Machine Learning — "
    "Supervised Learning (klasifikasi) & Unsupervised Learning (clustering)."
)

st.divider()

# ---------------- INTRO ----------------
st.markdown(
    """
    Selamat datang! Aplikasi ini memungkinkan Anda mengunggah dataset,
    memilih algoritma, mengatur hyperparameter, dan melihat hasil evaluasi
    secara langsung — tanpa menulis kode.

    👈 **Pilih menu di sidebar kiri** untuk mulai bereksperimen.
    """
)

# ---------------- CARD: DUA MODUL ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📊 Supervised Learning")
    st.markdown(
        """
        Klasifikasi dengan **Random Forest**, **SVM**, atau **KNN**.

        **Fitur:**
        - Upload dataset CSV / Excel
        - Pilih kolom target & kolom yang diabaikan
        - Atur proporsi data latih / uji
        - Hyperparameter via slider
        - Preview data latih & uji
        - Metrik: Accuracy, Precision, Recall, F1-Score
        - Confusion Matrix

        **Contoh dataset:** `social_network_ads.csv` (target: `Purchased`)
        """
    )

with col2:
    st.subheader("🔵 Unsupervised Learning")
    st.markdown(
        """
        Clustering dengan **K-Means** atau **DBSCAN**.

        **Fitur:**
        - Upload dataset CSV / Excel
        - Pilih kolom fitur (multi-select)
        - Atur jumlah cluster / `eps` / `min_samples`
        - Silhouette Score
        - Distribusi anggota cluster
        - Scatter plot PCA 2D

        **Contoh dataset:** `instagram_visits.csv`
        """
    )

st.divider()

# ---------------- QUICK GUIDE ----------------
st.subheader("🚀 Cara Cepat Memulai")

st.markdown(
    """
    1. **Siapkan dataset** dalam format `.csv`, `.xlsx`, atau `.xls`.
    2. **Pilih halaman** di sidebar kiri:
       - *Supervised Learning* → untuk klasifikasi
       - *Unsupervised Learning* → untuk clustering
    3. **Upload dataset** lewat file uploader di sidebar.
    4. **Atur parameter** (target, fitur, algoritma, hyperparameter).
    5. **Klik tombol** "Mulai Training" / "Mulai Clustering".
    6. **Lihat hasil** evaluasi & visualisasi di panel utama.
    """
)

# ---------------- TIPS ----------------
st.subheader("💡 Tips")

st.info(
    "**Kolom ID** (misalnya `User ID`) sebaiknya dimasukkan ke "
    "**Kolom yang Diabaikan** — kolom ID tidak memiliki makna prediktif "
    "dan justru bisa menurunkan performa model."
)

st.warning(
    "**Untuk Unsupervised Learning**, pilih minimal **2 kolom fitur** agar "
    "visualisasi PCA 2D menampilkan hasil yang optimal."
)

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Dibuat untuk **Tugas Proyek-3 — Implementasi Antarmuka Model Machine Learning "
    "Menggunakan Streamlit**. "
    "Lihat `README.md` untuk panduan lengkap & cara deploy."
)