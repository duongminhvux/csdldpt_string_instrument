from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASET_ROOT = DATA_DIR / "dataset"
DATA_ALL_ROOT = DATA_DIR / "data_all"
QUERY_ROOT = DATA_DIR / "query"

RESULTS_DIR = BASE_DIR / "results"
TOP5_WAV_DIR = RESULTS_DIR / "top5_wav"
FEATURES_CSV = RESULTS_DIR / "features.csv"
TOP5_RESULTS_CSV = RESULTS_DIR / "top5_results.csv"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "1233",
    "database": "string_instrument_search",
}