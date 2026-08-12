"""
Application constants.

This module contains project-wide constants that are unlikely to change.
Avoid hardcoding paths, filenames, or magic values elsewhere in the project.
"""

from pathlib import Path

# =============================================================================
# Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# Project Directories
# =============================================================================

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
GOLD_DATA_DIR = DATA_DIR / "gold"

LOG_DIR = PROJECT_ROOT / "logs"

MODEL_DIR = PROJECT_ROOT / "models"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

TEST_DIR = PROJECT_ROOT / "tests"

REPORT_DIR = PROJECT_ROOT / "reports"

# =============================================================================
# Configuration Files
# =============================================================================

CONFIG_FILE = CONFIG_DIR / "config.yaml"
PATHS_FILE = CONFIG_DIR / "paths.yaml"
MODELS_FILE = CONFIG_DIR / "models.yaml"
LOGGING_FILE = CONFIG_DIR / "logging.yaml"

# =============================================================================
# Log Files
# =============================================================================

DEFAULT_LOG_FILE = LOG_DIR / "browser_behavior.log"

# =============================================================================
# Chrome
# =============================================================================

CHROME_HISTORY_DATABASE_NAME = "History"

CHROME_PROFILE_DEFAULT = "Default"


# =============================================================================
# Randomness
# =============================================================================

DEFAULT_RANDOM_SEED = 42

# =============================================================================
# Machine Learning
# =============================================================================

DEFAULT_TEST_SIZE = 0.2

DEFAULT_VALIDATION_SIZE = 0.2

DEFAULT_BATCH_SIZE = 32

# =============================================================================
# MLflow
# =============================================================================

DEFAULT_MLFLOW_TRACKING_URI = "mlruns"

MLFLOW_CLUSTERING_EXPERIMENT = "browser-behavior-clustering"

MLFLOW_LSTM_EXPERIMENT = "browser-behavior-lstm"

MLFLOW_LSTM_TUNING_EXPERIMENT = "browser-behavior-lstm-tuning"

MLFLOW_RECOMMENDATION_EXPERIMENT = "browser-behavior-recommendation"

# =============================================================================
# Reports
# =============================================================================

CLUSTERING_REPORT_DIR = REPORT_DIR / "clustering"

LSTM_REPORT_DIR = REPORT_DIR / "lstm"

RECOMMENDATION_REPORT_DIR = REPORT_DIR / "recommendation"

# =============================================================================
# Streamlit
# =============================================================================

DEFAULT_HOST = "0.0.0.0"

DEFAULT_PORT = 8501

# =============================================================================
# Databricks
# =============================================================================

BRONZE_TABLE = "browser_bronze"

SILVER_TABLE = "browser_silver"

GOLD_TABLE = "browser_gold"

# =============================================================================
# Recommendation Engine
# =============================================================================

DEFAULT_TOP_K = 5

# =============================================================================
# Encoding
# =============================================================================

DEFAULT_ENCODING = "utf-8"

# =============================================================================
# Environment
# =============================================================================

DEVELOPMENT = "development"
PRODUCTION = "production"
TESTING = "testing"
