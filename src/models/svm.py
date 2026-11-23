import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from src.config import RANDOM_STATE
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


def run_svm(
    c_values: list[float] | None = None,
    gamma_values: list[str | float] | None = None,
) -> dict:
    if c_values is None:
        c_values = [0.1, 1.0, 10.0]
    if gamma_values is None:
        gamma_values = ["scale", "auto"]

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    best_score = -1.0
    best_cfg = None
    best_model = None

    # Combinaciones: kernel lineal (gamma no aplica) y kernel RBF (con C y gamma)
    candidates = [("linear", c, "scale") for c in c_values] + [
        ("rbf", c, g) for c in c_values for g in gamma_values
    ]

    for kernel, c_val, gamma_val in candidates:
        clf = SVC(
            kernel=kernel,
            C=c_val,
            gamma=gamma_val,
            probability=True,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)

        val_proba = clf.predict_proba(X_val)[:, 1]
        val_pred = clf.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_pred, val_proba)

        # Seleccionamos por F1 en validación (adecuado ante clases desbalanceadas)
        score = val_metrics["f1"]
        if score > best_score:
            best_score = score
            best_cfg = {"kernel": kernel, "C": c_val, "gamma": gamma_val}
            best_model = clf

    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = best_model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_pred, test_proba)

    print("=== SVM (Mejor en Val) ===")
    print("Mejor configuración:", best_cfg, f"| Val F1: {best_score:.4f}")
    print(
        f"Test -> Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | "
        f"Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}"
    )
    print("Matriz de confusión (Test):\n", test_metrics["confusion_matrix"])
    return {"model": best_model, "best_params": best_cfg, "test": test_metrics}


if __name__ == "__main__":
    run_svm()