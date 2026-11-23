import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import RANDOM_STATE, RESULTS_FIGURES_DIR
from src.models.baseline import load_splits


def get_fitted_models(X_train, y_train):
    models = {
        "LogReg (Baseline)": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            early_stopping=True,
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }
    for clf in models.values():
        clf.fit(X_train, y_train)
    return models


def main() -> None:
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_train, _, X_test, y_train, _, y_test = load_splits()
    models = get_fitted_models(X_train, y_train)

    # 1. Curva ROC y Curva Precisión-Recall
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, clf in models.items():
        y_proba = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_val = roc_auc_score(y_test, y_proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_val = average_precision_score(y_test, y_proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("Curva ROC (Test)")
    axes[0].set_xlabel("Tasa de Falsos Positivos")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (Recall)")
    axes[0].legend(loc="lower right")

    baseline_pos = np.mean(y_test)
    axes[1].axhline(
        baseline_pos,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Base ({baseline_pos:.3f})",
    )
    axes[1].set_title("Curva Precisión-Recall (Test)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precisión")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / "roc_pr_curves.png", dpi=300)
    plt.close()

    # 2. Matrices de confusión
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, clf) in zip(axes.ravel(), models.items()):
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["no (0)", "yes (1)"]
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close()

    print("Gráficas de resultados guardadas en:", RESULTS_FIGURES_DIR)


if __name__ == "__main__":
    main()