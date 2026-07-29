"""
Custom exceptions for the Browser Behavior Analytics project.
"""


class BrowserBehaviorAnalyticsError(Exception):
    """
    Base exception for the application.
    """


# =============================================================================
# Configuration
# =============================================================================

class ConfigurationError(BrowserBehaviorAnalyticsError):
    """
    Raised when configuration is invalid.
    """


class ConfigurationFileNotFoundError(ConfigurationError):
    """
    Raised when a configuration file cannot be found.
    """


# =============================================================================
# Data Collection
# =============================================================================

class DataCollectionError(BrowserBehaviorAnalyticsError):
    """
    Raised when data collection fails.
    """


class ChromeHistoryError(DataCollectionError):
    """
    Raised for Chrome History database errors.
    """


class HistoryDatabaseLockedError(ChromeHistoryError):
    """
    Raised when the Chrome History database is locked.
    """


class RAMCollectionError(DataCollectionError):
    """
    Raised when RAM metrics cannot be collected.
    """


# =============================================================================
# File System
# =============================================================================

class FileManagerError(BrowserBehaviorAnalyticsError):
    """
    Raised for file management errors.
    """


class DirectoryCreationError(FileManagerError):
    """
    Raised when a directory cannot be created.
    """


class FileCopyError(FileManagerError):
    """
    Raised when copying a file fails.
    """


# =============================================================================
# Data Processing
# =============================================================================

class DataValidationError(BrowserBehaviorAnalyticsError):
    """
    Raised when input data fails validation.
    """


class FeatureEngineeringError(BrowserBehaviorAnalyticsError):
    """
    Raised during feature engineering.
    """


# =============================================================================
# Machine Learning
# =============================================================================

class ModelTrainingError(BrowserBehaviorAnalyticsError):
    """
    Raised during model training.
    """


class PredictionError(BrowserBehaviorAnalyticsError):
    """
    Raised during inference.
    """