import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.supervised import (
    load_data,
    preprocess,
    split_data,
    build_model,
    train_and_evaluate,
    plot_confusion_matrix,
)

st.set_page_config(page_title="Supervised Learning", layout="wide")
st.title("Supervised Learning")
st.caption("Klasifikasi dengan Random Forest / SVM / KNN")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Pengaturan")

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
    )

    test_size = st.slider(
        "Test Size (%)", min_value=10, max_value=50, value=20, step=5
    )

    algorithm = st.selectbox(
        "Algoritma", ["Random Forest", "SVM", "KNN"]
    )

    params = {}
    if algorithm == "Random Forest":
        params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10)
        params["max_depth"] = st.slider("max_depth", 1, 30, 5, 1)
    elif algorithm == "SVM":
        params["C"] = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.01)
        params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    elif algorithm == "KNN":
        params["n_neighbors"] = st.slider("n_neighbors (K)", 1, 25, 5, 1)

    target_col = None
    ignore_cols = []
    train_btn = False

# ---------------- MAIN PANEL ----------------
if uploaded_file is None:
    st.info("Silakan upload dataset CSV atau Excel terlebih dahulu di sidebar.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

with st.sidebar:
    target_col = st.selectbox("Kolom Target", df.columns.tolist())
    ignore_cols = st.multiselect(
        "Kolom yang Diabaikan (opsional)",
        [c for c in df.columns if c != target_col],
        default=[],
    )
    train_btn = st.button("Mulai Training", use_container_width=True, type="primary")

# Cuplikan dataset
st.subheader("Cuplikan Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Baris", df.shape[0])
c2.metric("Jumlah Kolom", df.shape[1])
c3.metric("Kolom Target", target_col)

st.dataframe(df.head(10), use_container_width=True)

# Proses training
if train_btn:
    with st.spinner("Melatih model..."):
        try:
            X, y = preprocess(df, target_col, ignore_cols)
            X_train, X_test, y_train, y_test = split_data(X, y, test_size / 100)

            model = build_model(algorithm, params)
            metrics, cm = train_and_evaluate(
                model, X_train, X_test, y_train, y_test
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat training: {e}")
            st.stop()

    st.success(f"Training selesai menggunakan {algorithm}.")

    # ---------- SAMPEL DATA LATIH & UJI (sebelum evaluasi) ----------
    st.subheader("Sampel Data Latih & Uji")
    st.caption(
        f"Proporsi split: {100 - test_size}% latih / {test_size}% uji "
        f"(stratified berdasarkan target)."
    )

    col_train, col_test = st.columns(2)
    with col_train:
        st.markdown(f"**Data Latih** — {X_train.shape[0]} baris × {X_train.shape[1]} fitur")
        st.dataframe(
            X_train.head(10).assign(**{target_col: y_train.head(10).values}),
            use_container_width=True,
        )

    with col_test:
        st.markdown(f"**Data Uji** — {X_test.shape[0]} baris × {X_test.shape[1]} fitur")
        st.dataframe(
            X_test.head(10).assign(**{target_col: y_test.head(10).values}),
            use_container_width=True,
        )

    # ---------- HASIL EVALUASI ----------
    st.subheader("Hasil Evaluasi")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    m2.metric("Precision", f"{metrics['Precision']:.4f}")
    m3.metric("Recall", f"{metrics['Recall']:.4f}")
    m4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")

    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(cm)
    st.pyplot(fig)

    st.subheader("Pembagian Data")
    d1, d2 = st.columns(2)
    d1.info(f"Data Latih: {X_train.shape[0]} baris")
    d2.info(f"Data Uji: {X_test.shape[0]} baris")