from sklearn.metrics import f1_score
from sklearn.svm import SVC

from src.config import RANDOM_STATE
from src.models.baseline import load_splits


def main() -> None:
    X_train, X_val, _, y_train, y_val, _ = load_splits()
    best_score = -1.0
    best_cfg = None

    for kernel in ["linear", "rbf"]:
        for c_val in [0.1, 1.0, 10.0]:
            clf = SVC(
                kernel=kernel,
                C=c_val,
                gamma="scale",
                probability=True,
                random_state=RANDOM_STATE,
            )
            clf.fit(X_train, y_train)
            val_pred = clf.predict(X_val)
            score = f1_score(y_val, val_pred)
            if score > best_score:
                best_score = score
                best_cfg = {"kernel": kernel, "C": c_val, "gamma": "scale"}

    print("=== SVM (Mejor en Val) ===")
    print(f"Mejor configuración: {best_cfg} | Val F1: {best_score:.4f}")


if __name__ == "__main__":
    main()
