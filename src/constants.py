"""
Application constants.

Constants that are unlikely to change and are used throughout
the application belong here.
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
TEST_DIR = PROJECT_ROOT / "tests"

# =============================================================================
# Data Directories
# =============================================================================

RAW_DATA_DIR = DATA_DIR / "raw"
BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
GOLD_DATA_DIR = DATA_DIR / "gold"

# =============================================================================
# Chrome
# =============================================================================

CHROME_HISTORY_DATABASE = "History"

# =============================================================================
# File Extensions
# =============================================================================

CSV_EXTENSION = ".csv"
JSON_EXTENSION = ".json"
YAML_EXTENSION = ".yaml"

# =============================================================================
# Time
# =============================================================================

SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24

# =============================================================================
# Logging
# =============================================================================

DEFAULT_LOG_FILE = "project.log"