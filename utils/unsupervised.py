import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_data(uploaded_file):
    """Baca file CSV atau Excel menjadi DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Format file tidak didukung.")


def preprocess_features(df, feature_cols, ignore_cols=None):
    """Ambil fitur, encode kategorikal, dan lakukan scaling."""
    df = df.copy().dropna()

    # Guard: kolom yang dipilih sebagai fitur tidak boleh ikut dibuang
    if ignore_cols:
        ignore_cols = set(ignore_cols) - set(feature_cols)
        df = df.drop(columns=[c for c in ignore_cols if c in df.columns])

    X = df[feature_cols].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if len(X) == 0:
        raise ValueError("Tidak ada baris valid setelah pembersihan data.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled


def build_cluster_model(algorithm, params):
    if algorithm == "K-Means":
        return KMeans(
            n_clusters=params.get("n_clusters", 3),
            random_state=42,
            n_init=10,
        )
    if algorithm == "DBSCAN":
        return DBSCAN(
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5),
        )
    raise ValueError("Algoritma tidak dikenali.")


def run_clustering(model, X_scaled):
    labels = model.fit_predict(X_scaled)

    unique_labels = set(labels)
    if 1 < len(unique_labels) < len(X_scaled):
        score = silhouette_score(X_scaled, labels)
    else:
        score = None

    return labels, score


def plot_clusters(X_scaled, labels):
    """Visualisasi hasil clustering dalam 2D memakai PCA.

    Aman untuk 1 fitur (padding kolom kedua dengan 0) maupun
    >= 2 fitur (reduksi normal ke 2 komponen).
    """
    n_comp = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    # Kalau cuma 1 fitur, tambah kolom dummy berisi 0 supaya scatter 2D tetap jalan
    if n_comp == 1:
        X_2d = np.hstack([X_2d, np.zeros((X_2d.shape[0], 1))])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="tab10", s=80, alpha=0.85,
        edgecolors="white", linewidths=0.8,
    )
    ax.set_xlabel("Fitur 1 (PCA)")
    ax.set_ylabel("Fitur 2 (PCA)" if n_comp == 2 else "(padding)")
    ax.set_title("Scatter Plot Hasil Clustering")
    ax.grid(True, alpha=0.3)
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.tight_layout()
    return fig