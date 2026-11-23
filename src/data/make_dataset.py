import io
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CSV_SEP,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from src.data.preprocess import build_preprocessor, clean_bank_data


def _load_raw_file(path, sep: str = ";") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo en: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw_excel = pd.read_excel(path, sheet_name=0)
        if raw_excel.shape[1] == 1:
            col = raw_excel.iloc[:, 0].astype(str)
            csv_buffer = io.StringIO(f"{col.name}\n" + "\n".join(col.tolist()))
            return pd.read_csv(csv_buffer, sep=sep)
        return raw_excel

    return pd.read_csv(path, sep=sep)


def _to_df(matrix: np.ndarray, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, columns=columns)


def main() -> None:
    if not np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1.0):
        raise ValueError("TRAIN_SIZE + VAL_SIZE + TEST_SIZE debe sumar 1.0.")

    np.random.seed(RANDOM_STATE)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = _load_raw_file(RAW_DATA_PATH, sep=CSV_SEP)
    df_clean = clean_bank_data(df_raw, target_col=TARGET_COL)

    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    # 1) División estratificada: separar test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2) Dividir el resto (85%) en train (70% del total) y val (15% del total)
    val_ratio_from_temp = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_from_temp,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # 3) Ajustar preprocesador SOLO en X_train y transformar train, val y test
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    _to_df(X_train_proc, feature_names).to_csv(
        PROCESSED_DATA_DIR / "X_train.csv", index=False
    )
    _to_df(X_val_proc, feature_names).to_csv(
        PROCESSED_DATA_DIR / "X_val.csv", index=False
    )
    _to_df(X_test_proc, feature_names).to_csv(
        PROCESSED_DATA_DIR / "X_test.csv", index=False
    )

    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    joblib.dump(preprocessor, PROCESSED_DATA_DIR / "preprocessor.joblib")

    print("Procesamiento completado con semilla:", RANDOM_STATE)
    print(f"Train: {X_train_proc.shape} | Val: {X_val_proc.shape} | Test: {X_test_proc.shape}")
    print(
        f"Proporción clase 1 ('yes') -> train: {y_train.mean():.4f}, "
        f"val: {y_val.mean():.4f}, test: {y_test.mean():.4f}"
    )


if __name__ == "__main__":
    main()