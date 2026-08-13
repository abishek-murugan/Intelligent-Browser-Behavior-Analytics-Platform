"""
MLflow experiment helpers.

Centralizes MLflow tracking URI resolution and experiment setup so
all pipeline stages (clustering, LSTM, recommendation) log to a
consistent backend. The tracking URI can be overridden at runtime
through the ``MLFLOW_TRACKING_URI`` environment variable, which is
useful when switching to a remote tracking server (e.g. Azure ML).
"""

from __future__ import annotations

import os
from typing import Any

import mlflow

from src.constants import (
    DEFAULT_MLFLOW_TRACKING_URI,
    MLFLOW_CLUSTERING_EXPERIMENT,
    MLFLOW_LSTM_EXPERIMENT,
    MLFLOW_LSTM_TUNING_EXPERIMENT,
    MLFLOW_RECOMMENDATION_EXPERIMENT,
    PROJECT_ROOT,
)
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAMES = {
    "clustering": MLFLOW_CLUSTERING_EXPERIMENT,
    "lstm": MLFLOW_LSTM_EXPERIMENT,
    "lstm_tuning": MLFLOW_LSTM_TUNING_EXPERIMENT,
    "recommendation": MLFLOW_RECOMMENDATION_EXPERIMENT,
}


def get_tracking_uri() -> str:
    """
    Resolve the MLflow tracking URI.

    Priority: the ``MLFLOW_TRACKING_URI`` environment variable, then
    the configured value, then the local default. Ensures relative paths
    are resolved absolutely against PROJECT_ROOT to prevent directory fragmentation.
    """

    raw_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        get_config()
        .get("mlflow", {})
        .get(
            "tracking_uri",
            DEFAULT_MLFLOW_TRACKING_URI,
        ),
    )

    if not raw_uri or raw_uri == "mlruns":
        return (PROJECT_ROOT / "mlruns").as_uri()

    if "://" not in raw_uri and not os.path.isabs(raw_uri):
        return (PROJECT_ROOT / raw_uri).as_uri()

    return raw_uri


def setup_mlflow() -> None:
    """Configure the MLflow tracking backend for this process."""

    tracking_uri = get_tracking_uri()

    if _uses_file_store(tracking_uri):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(tracking_uri)

    logger.info("MLflow tracking URI: %s", tracking_uri)


def _uses_file_store(tracking_uri: str) -> bool:
    """Return whether the tracking URI targets the local file store."""

    if "://" not in tracking_uri:
        return True

    return tracking_uri.startswith("file://")


def get_experiment_id(name: str) -> str:
    """
    Return the MLflow experiment id for a named experiment, creating
    it if it does not exist.

    Parameters
    ----------
    name:
        Experiment name.

    Returns
    -------
    str
        The experiment id.
    """

    experiment = mlflow.get_experiment_by_name(name)

    if experiment is not None:
        return experiment.experiment_id

    logger.info("Creating MLflow experiment: %s", name)

    return mlflow.create_experiment(name)


def get_experiment_name(key: str) -> str:
    """
    Return the configured experiment name for a pipeline key.
    """

    return EXPERIMENT_NAMES[key]


def log_artifact_batch(
    paths: list[str | os.PathLike[str]],
) -> None:
    """
    Log a batch of local artifact paths to the active MLflow run.
    """

    for path in paths:
        mlflow.log_artifact(str(path))

        logger.debug("Logged MLflow artifact: %s", path)


def log_dict(dictionary: dict[str, Any], key: str) -> None:
    """
    Log a dictionary as a JSON artifact on the active MLflow run.
    """

    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
    ) as file:
        json.dump(dictionary, file, indent=2, default=str)
        artifact_path = file.name

    mlflow.log_artifact(artifact_path, artifact_path=key)

    os.unlink(artifact_path)


__all__ = [
    "EXPERIMENT_NAMES",
    "get_experiment_id",
    "get_experiment_name",
    "get_tracking_uri",
    "log_artifact_batch",
    "log_dict",
    "setup_mlflow",
]
