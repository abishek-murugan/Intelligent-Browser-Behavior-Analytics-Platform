"""
Custom exceptions for the Browser Behavior Analytics project.

All project-specific exceptions inherit from
BrowserBehaviorAnalyticsError.
"""


# =============================================================================
# Base Exception
# =============================================================================

class BrowserBehaviorAnalyticsError(Exception):
    """Base exception for the application."""


# =============================================================================
# Configuration
# =============================================================================

class ConfigurationError(BrowserBehaviorAnalyticsError):
    """Raised when a configuration is invalid."""


class ConfigurationFileNotFoundError(ConfigurationError):
    """Raised when a configuration file cannot be found."""


# =============================================================================
# File System
# =============================================================================

class FileManagerError(BrowserBehaviorAnalyticsError):
    """Raised for file system related errors."""


class DirectoryCreationError(FileManagerError):
    """Raised when creating a directory fails."""


class FileCopyError(FileManagerError):
    """Raised when copying a file fails."""


class FileMoveError(FileManagerError):
    """Raised when moving a file fails."""


class FileDeletionError(FileManagerError):
    """Raised when deleting a file fails."""


class FileReadError(FileManagerError):
    """Raised when reading a file fails."""


class FileWriteError(FileManagerError):
    """Raised when writing a file fails."""


# =============================================================================
# Data Ingestion
# =============================================================================

class DataCollectionError(BrowserBehaviorAnalyticsError):
    """Raised during data collection."""


class ChromeHistoryError(DataCollectionError):
    """Raised for Chrome History database errors."""


class HistoryDatabaseLockedError(ChromeHistoryError):
    """Raised when the Chrome History database is locked."""


class BrowserHistoryExtractionError(ChromeHistoryError):
    """Raised when browser history extraction fails."""


class RAMCollectionError(DataCollectionError):
    """Raised when RAM usage collection fails."""


class AppUsageCollectionError(DataCollectionError):
    """Raised when application usage collection fails."""


# =============================================================================
# Data Processing
# =============================================================================

class DataValidationError(BrowserBehaviorAnalyticsError):
    """Raised when input data fails validation."""


class DataCleaningError(BrowserBehaviorAnalyticsError):
    """Raised during data cleaning."""


class FeatureEngineeringError(BrowserBehaviorAnalyticsError):
    """Raised during feature engineering."""


class DataTransformationError(BrowserBehaviorAnalyticsError):
    """Raised during data transformation."""


# =============================================================================
# Databricks
# =============================================================================

class DatabricksError(BrowserBehaviorAnalyticsError):
    """Base exception for Databricks operations."""


class DatabricksConnectionError(DatabricksError):
    """Raised when Databricks connection fails."""


class DeltaTableError(DatabricksError):
    """Raised during Delta Lake operations."""


class SparkSessionError(DatabricksError):
    """Raised when Spark session creation fails."""


# =============================================================================
# Machine Learning
# =============================================================================

class ModelTrainingError(BrowserBehaviorAnalyticsError):
    """Raised during model training."""


class ModelLoadingError(BrowserBehaviorAnalyticsError):
    """Raised when loading a trained model fails."""


class PredictionError(BrowserBehaviorAnalyticsError):
    """Raised during model inference."""


class ModelSerializationError(BrowserBehaviorAnalyticsError):
    """Raised when saving or loading model artifacts."""


# =============================================================================
# Deep Learning
# =============================================================================

class DeepLearningError(BrowserBehaviorAnalyticsError):
    """Base exception for deep learning."""


class SequenceGenerationError(DeepLearningError):
    """Raised while generating training sequences."""


class LSTMTrainingError(DeepLearningError):
    """Raised during LSTM training."""


class RNNTrainingError(DeepLearningError):
    """Raised during RNN training."""


# =============================================================================
# Recommendation Engine
# =============================================================================

class RecommendationEngineError(BrowserBehaviorAnalyticsError):
    """Raised when recommendation generation fails."""


# =============================================================================
# Visualization
# =============================================================================

class VisualizationError(BrowserBehaviorAnalyticsError):
    """Raised during chart or dashboard generation."""


# =============================================================================
# Deployment
# =============================================================================

class DeploymentError(BrowserBehaviorAnalyticsError):
    """Raised during deployment."""


# =============================================================================
# External Services
# =============================================================================

class ExternalServiceError(BrowserBehaviorAnalyticsError):
    """Raised when communicating with an external service."""
