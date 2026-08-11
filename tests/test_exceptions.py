from src.exceptions import (
    AppUsageCollectionError,
    BrowserBehaviorAnalyticsError,
    BrowserHistoryExtractionError,
    ChromeHistoryError,
    ConfigurationError,
    ConfigurationFileNotFoundError,
    DatabricksConnectionError,
    DatabricksError,
    DataCleaningError,
    DataCollectionError,
    DataTransformationError,
    DataValidationError,
    DeepLearningError,
    DeltaTableError,
    DeploymentError,
    DirectoryCreationError,
    ExternalServiceError,
    FeatureEngineeringError,
    FileCopyError,
    FileDeletionError,
    FileManagerError,
    FileMoveError,
    FileReadError,
    FileWriteError,
    HistoryDatabaseLockedError,
    LSTMTrainingError,
    ModelLoadingError,
    ModelSerializationError,
    ModelTrainingError,
    PredictionError,
    RAMCollectionError,
    RecommendationEngineError,
    RNNTrainingError,
    SequenceGenerationError,
    SparkSessionError,
    VisualizationError,
)


def test_base_exception():
    exc = BrowserBehaviorAnalyticsError("boom")

    assert isinstance(exc, Exception)
    assert str(exc) == "boom"


def test_all_exceptions_derive_from_base():
    exceptions = [
        ConfigurationError,
        ConfigurationFileNotFoundError,
        FileManagerError,
        DirectoryCreationError,
        FileCopyError,
        FileMoveError,
        FileDeletionError,
        FileReadError,
        FileWriteError,
        DataCollectionError,
        ChromeHistoryError,
        HistoryDatabaseLockedError,
        BrowserHistoryExtractionError,
        RAMCollectionError,
        AppUsageCollectionError,
        DataValidationError,
        DataCleaningError,
        FeatureEngineeringError,
        DataTransformationError,
        DatabricksError,
        DatabricksConnectionError,
        DeltaTableError,
        SparkSessionError,
        ModelTrainingError,
        ModelLoadingError,
        PredictionError,
        ModelSerializationError,
        DeepLearningError,
        SequenceGenerationError,
        LSTMTrainingError,
        RNNTrainingError,
        RecommendationEngineError,
        VisualizationError,
        DeploymentError,
        ExternalServiceError,
    ]

    for exc_type in exceptions:
        assert issubclass(exc_type, BrowserBehaviorAnalyticsError), exc_type


def test_exception_hierarchy():
    assert issubclass(ConfigurationFileNotFoundError, ConfigurationError)
    assert issubclass(HistoryDatabaseLockedError, ChromeHistoryError)
    assert issubclass(BrowserHistoryExtractionError, ChromeHistoryError)
    assert issubclass(ChromeHistoryError, DataCollectionError)
    assert issubclass(RAMCollectionError, DataCollectionError)
    assert issubclass(AppUsageCollectionError, DataCollectionError)
    assert issubclass(DirectoryCreationError, FileManagerError)
    assert issubclass(FileCopyError, FileManagerError)
    assert issubclass(FileMoveError, FileManagerError)
    assert issubclass(FileDeletionError, FileManagerError)
    assert issubclass(FileReadError, FileManagerError)
    assert issubclass(FileWriteError, FileManagerError)
    assert issubclass(DatabricksConnectionError, DatabricksError)
    assert issubclass(DeltaTableError, DatabricksError)
    assert issubclass(SparkSessionError, DatabricksError)
    assert issubclass(SequenceGenerationError, DeepLearningError)
    assert issubclass(LSTMTrainingError, DeepLearningError)
    assert issubclass(RNNTrainingError, DeepLearningError)
