import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(uploaded_file):
    """Baca file CSV atau Excel menjadi DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")


def preprocess(df, target_col, ignore_cols=None):
    """Bersihkan data, encode kolom kategorikal, pisahkan X dan y."""
    df = df.copy()

    # Guard: target_col tidak boleh ikut dibuang
    if ignore_cols:
        ignore_cols = set(ignore_cols) - {target_col}
        df = df.drop(columns=[c for c in ignore_cols if c in df.columns])

    # Drop baris tanpa target dulu, lalu drop baris dengan fitur kosong
    df = df.dropna(subset=[target_col])
    df = df.dropna()

    if len(df) == 0:
        raise ValueError("Tidak ada baris valid setelah pembersihan data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode fitur kategorikal
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target jika kategorikal
    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y


def split_data(X, y, test_size, random_state=42):
    """Split data dengan stratify; fallback tanpa stratify bila ada kelas langka."""
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )


def build_model(algorithm, params):
    """Buat model sesuai algoritma yang dipilih."""
    if algorithm == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=42,
        )
    if algorithm == "SVM":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            probability=True,
        )
    if algorithm == "KNN":
        return KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    raise ValueError("Algoritma tidak dikenali.")


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Scaling, latih, prediksi, dan hitung metrik."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    average = "binary" if len(np.unique(y_test)) == 2 else "weighted"

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig