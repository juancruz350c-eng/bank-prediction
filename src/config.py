from pathlib import Path

# Rutas principales del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"
RESULTS_FIGURES_DIR = FIGURES_DIR / "resultados"

# Archivo de entrada en data/raw/
RAW_FILENAME = "bank.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_FILENAME
CSV_SEP = ";"

# Semilla fija para reproducibilidad (división y prueba)
RANDOM_STATE = 42

# Proporciones de división estratificada (deben sumar 1.0)
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Columna objetivo ('yes' -> 1, 'no' -> 0)
TARGET_COL = "y"