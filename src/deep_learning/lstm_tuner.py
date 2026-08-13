"""Optuna hyperparameter tuning for the next-session PyTorch LSTM.

The search space, trial budget, and training schedule are declared in
``config/models.yaml`` (``deep_learning.lstm.tuning``) while the study
storage is configured in ``config/config.yaml`` (``optuna``). Every
trial reuses the exact training path of :class:`LSTMPipeline` and is
logged to the ``browser-behavior-lstm-tuning`` MLflow experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch

from src.constants import PROJECT_ROOT
from src.deep_learning.lstm_pipeline import LSTMPipeline
from src.exceptions import LSTMTuningError
from src.utils.config_loader import get_config, get_models, get_paths
from src.utils.logger import get_logger
from src.utils.mlflow_utils import get_experiment_id, get_experiment_name, setup_mlflow

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    """Outcome of a completed Optuna hyperparameter search."""

    study: optuna.study.Study
    best_params: dict[str, Any]
    best_value: float
    n_trials: int


class LSTMTuner:
    """Tune the multi-task LSTM with Optuna, logging trials to MLflow."""

    def __init__(
        self,
        input_path: str | Path | None = None,
        track_mlflow: bool = True,
        device: str | None = None,
        n_trials: int | None = None,
        timeout: float | None = None,
        n_jobs: int | None = None,
        seed: int | None = None,
        search_space: dict[str, Any] | None = None,
        study_name: str | None = None,
        storage: str | None = None,
    ) -> None:
        paths = get_paths()["paths"]
        models = get_models()["deep_learning"]["lstm"]
        config = get_config()

        tuning = models.get("tuning", {})
        optuna_config = config.get("optuna", {})

        self.input_path = Path(input_path or paths["behavior_sequences"]).expanduser()
        self.track_mlflow = track_mlflow
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_trials = int(n_trials if n_trials is not None else tuning.get("n_trials", 15))
        self.timeout = timeout if timeout is not None else tuning.get("timeout_seconds")
        self.n_jobs = int(n_jobs if n_jobs is not None else tuning.get("n_jobs", 1))
        self.seed = int(seed if seed is not None else config["project"].get("random_seed", 42))
        self.study_name = study_name or optuna_config.get("study_name", "lstm-next-session-tuning")
        self.storage = self._normalize_storage(
            storage or optuna_config.get("storage", "sqlite:///mlruns/optuna_lstm.db")
        )
        self.base_overrides = {
            "epochs": int(tuning.get("epochs", 25)),
            "early_stopping_patience": int(tuning.get("early_stopping_patience", 5)),
        }
        self.search_space = (
            search_space if search_space is not None else tuning.get("search_space", {})
        )

        self._frame: pd.DataFrame | None = None
        self._train: pd.DataFrame | None = None
        self._validation: pd.DataFrame | None = None
        self._category_classes: np.ndarray | None = None

    def run(self) -> TuningResult:
        """Load the gold sequences, optimize the search space, and log results."""
        self._prepare_data()

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
            sampler=sampler,
        )
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        if self.track_mlflow:
            self._log_study_summary(study)

        logger.info(
            "LSTM tuning completed | trials=%d | best_validation_loss=%.4f",
            len(study.trials),
            study.best_value,
        )
        logger.info("LSTM best parameters: %s", study.best_params)

        return TuningResult(
            study=study,
            best_params=study.best_params,
            best_value=float(study.best_value),
            n_trials=len(study.trials),
        )

    def _prepare_data(self) -> None:
        if not self.input_path.is_file():
            raise LSTMTuningError(f"Behavior sequences dataset not found: {self.input_path}")
        frame = pd.read_parquet(self.input_path)
        if frame.empty:
            raise LSTMTuningError(f"Behavior sequences dataset is empty: {self.input_path}")

        self._frame = frame.reset_index(drop=True)
        self._category_classes = frame["target_category"].astype(str).unique()
        splits = LSTMPipeline._split(frame)
        self._train = splits["train"]
        self._validation = splits["validation"]

        logger.info(
            "Tuning data prepared | train=%d | validation=%d | categories=%d",
            len(self._train),
            len(self._validation),
            len(self._category_classes),
        )

    def _objective(self, trial: optuna.trial.Trial) -> float:
        params = self._suggest_params(trial)
        pipeline = self._pipeline_for(params)
        _, history = pipeline._fit(self._train, self._validation)  # noqa: SLF001
        validation_loss = float(history[-1]["validation_loss"])

        if self.track_mlflow:
            self._log_trial(trial, params, validation_loss)

        logger.debug("Trial %d | loss=%.4f | params=%s", trial.number, validation_loss, params)
        return validation_loss

    def _pipeline_for(self, params: dict[str, Any]) -> LSTMPipeline:
        pipeline = LSTMPipeline(
            input_path=self.input_path,
            track_mlflow=False,
            device=self.device,
            overrides={**self.base_overrides, **params},
        )
        pipeline._category_classes = self._category_classes  # noqa: SLF001
        return pipeline

    def _suggest_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        if not self.search_space:
            raise LSTMTuningError("No Optuna search space configured in models.yaml.")
        params: dict[str, Any] = {}
        for name, bounds in self.search_space.items():
            low, high = bounds["min"], bounds["max"]
            if name == "learning_rate":
                params[name] = trial.suggest_float(name, low, high, log=True)
            elif isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)
        return params

    def _log_trial(
        self,
        trial: optuna.trial.Trial,
        params: dict[str, Any],
        validation_loss: float,
    ) -> None:
        import mlflow

        setup_mlflow()
        experiment_id = get_experiment_id(get_experiment_name("lstm_tuning"))
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"trial-{trial.number:02d}",
        ) as run:
            mlflow.log_params({**self.base_overrides, **params})
            mlflow.log_metric("validation_loss", validation_loss)
            trial.set_user_attr("mlflow_run_id", run.info.run_id)

    def _log_study_summary(self, study: optuna.study.Study) -> None:
        import tempfile

        import mlflow

        setup_mlflow()
        experiment_id = get_experiment_id(get_experiment_name("lstm_tuning"))
        with mlflow.start_run(experiment_id=experiment_id, run_name="study-summary") as run:
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_validation_loss", float(study.best_value))
            mlflow.log_metric("n_trials", len(study.trials))
            try:
                import optuna.visualization as visualization

                figure = visualization.plot_param_importances(study)
                with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as file:
                    figure.write_html(file.name)
                    mlflow.log_artifact(file.name, artifact_path="importance")
            except Exception as err:  # pragma: no cover - optional visualization
                logger.warning("Could not log Optuna importance plot: %s", err)
            logger.info("MLflow tuning summary run logged | run_id=%s", run.info.run_id)

    @staticmethod
    def _normalize_storage(storage: str) -> str:
        if storage.startswith("sqlite:///"):
            database = storage.removeprefix("sqlite:///")
            if not Path(database).is_absolute():
                database = str((PROJECT_ROOT / database).resolve())
            return f"sqlite:///{database}"
        return storage


def main() -> None:
    """Tune the LSTM, then train the final model with the best parameters."""
    tuner = LSTMTuner()
    result = tuner.run()
    logger.info("Training final LSTM with best parameters: %s", result.best_params)
    LSTMPipeline(overrides=result.best_params).run()


if __name__ == "__main__":
    main()
