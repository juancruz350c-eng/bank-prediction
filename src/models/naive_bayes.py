import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB

from src.config import PROCESSED_DATA_DIR
from src.models.baseline import load_splits


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


def run_naive_bayes(var_smoothing: float = 1e-9) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # GaussianNB es adecuado para variables numéricas continuas / estandarizadas y columnas one-hot densas
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = model.predict(X_val)
    val_metrics = evaluate_predictions(y_val, val_pred, val_proba)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_pred, test_proba)

    print("=== NAIVE BAYES (GaussianNB) ===")
    print(f"Val  -> F1: {val_metrics['f1']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(
        f"Test -> Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | "
        f"Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}"
    )
    print("Matriz de confusión (Test):\n", test_metrics["confusion_matrix"])
    return {"model": model, "val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    run_naive_bayes()