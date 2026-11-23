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
from sklearn.neural_network import MLPClassifier

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


def run_mlp(
    architectures: list[tuple[int, ...]] | None = None,
    alphas: list[float] | None = None,
    activation: str = "relu",
    max_iter: int = 500,
) -> dict:
    if architectures is None:
        architectures = [(64, 32), (128, 64)]
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2]

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    best_score = -1.0
    best_cfg = None
    best_model = None

    for hidden_layers in architectures:
        for alpha_val in alphas:
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver="adam",
                alpha=alpha_val,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                max_iter=max_iter,
                random_state=RANDOM_STATE,
            )
            mlp.fit(X_train, y_train)

            val_proba = mlp.predict_proba(X_val)[:, 1]
            val_pred = mlp.predict(X_val)
            val_metrics = evaluate_predictions(y_val, val_pred, val_proba)

            score = val_metrics["f1"]
            if score > best_score:
                best_score = score
                best_cfg = {
                    "hidden_layer_sizes": hidden_layers,
                    "activation": activation,
                    "alpha": alpha_val,
                    "early_stopping": True,
                }
                best_model = mlp

    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = best_model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_pred, test_proba)

    print("=== MLP (Perceptrón Multicapa) ===")
    print("Mejor configuración:", best_cfg, f"| Val F1: {best_score:.4f}")
    print(
        f"Test -> Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | "
        f"Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}"
    )
    print("Matriz de confusión (Test):\n", test_metrics["confusion_matrix"])
    return {"model": best_model, "best_params": best_cfg, "test": test_metrics}


if __name__ == "__main__":
    run_mlp()