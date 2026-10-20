import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.unsupervised import (
    load_data,
    preprocess_features,
    build_cluster_model,
    run_clustering,
    plot_clusters,
)

st.set_page_config(page_title="Unsupervised Learning", layout="wide")
st.title("Unsupervised Learning")
st.caption("Clustering dengan K-Means / DBSCAN")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Pengaturan")

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
    )

    algorithm = st.selectbox("Algoritma", ["K-Means", "DBSCAN"])

    params = {}
    if algorithm == "K-Means":
        params["n_clusters"] = st.slider("Jumlah Cluster (K)", 2, 10, 3, 1)
    else:
        params["eps"] = st.slider("eps", 0.1, 5.0, 0.5, 0.1)
        params["min_samples"] = st.slider("min_samples", 2, 20, 5, 1)

    feature_cols = []
    ignore_cols = []
    cluster_btn = False

# ---------------- MAIN PANEL ----------------
if uploaded_file is None:
    st.info("Silakan upload dataset CSV atau Excel terlebih dahulu di sidebar.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()

with st.sidebar:
    feature_cols = st.multiselect(
        "Kolom Fitur yang Dianalisis",
        df.columns.tolist(),
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
    )
    ignore_cols = st.multiselect(
        "Kolom yang Diabaikan (opsional)",
        [c for c in df.columns if c not in feature_cols],
        default=[],
    )
    cluster_btn = st.button("Mulai Clustering", use_container_width=True, type="primary")

st.subheader("Cuplikan Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Baris", df.shape[0])
c2.metric("Jumlah Kolom", df.shape[1])
c3.metric("Fitur Terpilih", len(feature_cols))

st.dataframe(df.head(10), use_container_width=True)

if cluster_btn:
    if len(feature_cols) < 1:
        st.warning("Pilih minimal 1 kolom fitur terlebih dahulu.")
        st.stop()

    with st.spinner("Menjalankan clustering..."):
        try:
            X, X_scaled = preprocess_features(df, feature_cols, ignore_cols)
            model = build_cluster_model(algorithm, params)
            labels, score = run_clustering(model, X_scaled)
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.stop()

    st.success(f"Clustering selesai menggunakan {algorithm}.")

    st.subheader("Hasil Clustering")
    a1, a2 = st.columns(2)
    a1.metric(
        "Silhouette Score",
        f"{score:.4f}" if score is not None else "N/A",
    )
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    a2.metric("Jumlah Cluster Terbentuk", n_clusters)

    st.subheader("Distribusi Anggota Cluster")
    dist = pd.Series(labels, name="Cluster").value_counts().sort_index()
    st.bar_chart(dist)

    st.subheader("Scatter Plot (PCA 2D)")
    fig = plot_clusters(X_scaled, labels)
    st.pyplot(fig)