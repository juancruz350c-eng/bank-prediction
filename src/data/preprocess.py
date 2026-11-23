import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_bank_data(df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
    """Limpia nombres de columnas, duplicados y convierte el objetivo a 0/1."""
    df = df.copy()
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}'.")

    df = df.drop_duplicates().dropna(subset=[target_col])

    # Limpiar comillas o espacios en columnas de texto
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    # Mapear 'yes'/'no' a 1/0
    target_map = {"yes": 1, "no": 0, "1": 1, "0": 0}
    mapped = df[target_col].str.lower().map(target_map)
    if mapped.isna().any():
        raise ValueError(f"Valores no reconocidos en '{target_col}': {df[target_col].unique()}")

    df[target_col] = mapped.astype(int)
    return df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Crea el transformador para numéricas (escalado) y categóricas (One-Hot)."""
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )