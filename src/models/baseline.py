import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE


def load_splits():
    X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
    X_val = np.load(PROCESSED_DATA_DIR / "X_val.npy")
    X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DATA_DIR / "y_val.npy")
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = clf.predict(X_val)
    print("=== BASELINE: Regresión Logística ===")
    print(
        f"Val  -> F1: {f1_score(y_val, val_pred):.4f} | "
        f"PR-AUC: {average_precision_score(y_val, val_proba):.4f} | "
        f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}"
    )


if __name__ == "__main__":
    main()
