from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB

from src.models.baseline import load_splits


def main() -> None:
    X_train, X_val, _, y_train, y_val, _ = load_splits()
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = clf.predict(X_val)
    print("=== NAIVE BAYES (GaussianNB) ===")
    print(
        f"Val  -> F1: {f1_score(y_val, val_pred):.4f} | "
        f"PR-AUC: {average_precision_score(y_val, val_proba):.4f} | "
        f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}"
    )


if __name__ == "__main__":
    main()
