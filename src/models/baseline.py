import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE


def load_splits():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").iloc[:, 0].values
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").iloc[:, 0].values
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").iloc[:, 0].values
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def run_baseline(C: float = 1.0, max_iter: int = 1000) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = model.predict(X_val)
    val_metrics = evaluate_predictions(y_val, val_pred, val_proba)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_pred, test_proba)

    print("=== BASELINE: Regresión Logística ===")
    print(f"Val  -> F1: {val_metrics['f1']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(
        f"Test -> Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | "
        f"Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}"
    )
    print("Matriz de confusión (Test):\n", test_metrics["confusion_matrix"])
    return {"model": model, "val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    run_baseline()