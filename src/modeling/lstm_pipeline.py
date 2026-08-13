"""Train, evaluate, save, and use the next-session PyTorch LSTM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.exceptions import DataValidationError, FileReadError, LSTMTrainingError, PredictionError
from src.modeling.dataset_builder import DatasetBuilder
from src.modeling.lstm_model import NextSessionLSTM
from src.utils.config_loader import get_config, get_models, get_paths
from src.utils.logger import get_logger
from src.utils.mlflow_utils import get_experiment_id, get_experiment_name, setup_mlflow

logger = get_logger(__name__)


@dataclass
class LSTMArtifacts:
    """Model and preprocessors needed for deterministic inference."""

    model: NextSessionLSTM
    scaler: StandardScaler
    encoder: LabelEncoder
    feature_columns: list[str]
    sequence_length: int


class LSTMPipeline:
    """Chronologically train a multi-task LSTM on sliding session windows."""

    REQUIRED_COLUMNS = {
        "feature_vectors",
        "target_features",
        "target_category",
        "target_session_id",
    }

    def __init__(
        self,
        input_path: str | Path | None = None,
        model_path: str | Path | None = None,
        predictions_path: str | Path | None = None,
        track_mlflow: bool = True,
        device: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        paths = get_paths()["paths"]
        self.settings = get_models()["deep_learning"]["lstm"].copy()
        self.settings.update(overrides or {})
        self.input_path = Path(input_path or paths["behavior_sequences"]).expanduser()
        self.model_path = Path(model_path or paths["lstm_model"]).expanduser()
        self.predictions_path = Path(predictions_path or paths["lstm_predictions"]).expanduser()
        self.track_mlflow = track_mlflow
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.artifacts: LSTMArtifacts | None = None

    def run(self) -> dict[str, Any]:
        """Train chronologically, persist artifacts, and predict the held-out sessions."""
        frame = self._load_sequences()
        # The fixed category vocabulary is metadata, not a learned target value;
        # keeping it complete lets chronologically held-out categories be scored.
        self._category_classes = frame["target_category"].astype(str).unique()
        self._seed_everything(get_config()["project"].get("random_seed", 42))
        splits = self._split(frame)
        artifacts, history = self._fit(splits["train"], splits["validation"])
        self.artifacts = artifacts
        metrics, predictions = self.evaluate(splits["test"])
        self.save()
        self._save_predictions(predictions)
        run_id = self._log_mlflow(metrics, history, len(frame))
        logger.info(
            "LSTM pipeline completed | test_samples=%d | rmse=%.4f",
            len(predictions),
            metrics["rmse"],
        )
        return {
            "metrics": metrics,
            "predictions": predictions,
            "history": history,
            "run_id": run_id,
        }

    def evaluate(self, frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
        """Evaluate the fitted model and return readable per-session predictions."""
        if self.artifacts is None:
            raise PredictionError("Fit or load the LSTM model before evaluation.")
        sequences, targets, categories = self._arrays(frame, fit=False)
        self.artifacts.model.eval()
        with torch.no_grad():
            regressions, logits = self.artifacts.model(
                torch.tensor(sequences, dtype=torch.float32, device=self.device)
            )
        predicted_scaled = regressions.cpu().numpy()
        predicted = self.artifacts.scaler.inverse_transform(predicted_scaled)
        actual = self.artifacts.scaler.inverse_transform(targets)
        category_ids = logits.argmax(dim=1).cpu().numpy()
        labels = self.artifacts.encoder.inverse_transform(category_ids)
        metrics = {
            "mse": float(mean_squared_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "mae": float(mean_absolute_error(actual, predicted)),
            "category_accuracy": float(accuracy_score(categories, labels)),
        }
        result = pd.DataFrame(
            {
                "target_session_id": frame["target_session_id"].to_numpy(),
                "actual_category": categories,
                "predicted_category": labels,
                "category_correct": categories == labels,
            }
        )
        for index, name in enumerate(self.artifacts.feature_columns):
            result[f"actual_{name}"] = actual[:, index]
            result[f"predicted_{name}"] = predicted[:, index]
        return metrics, result

    def predict_next(self, session_windows: np.ndarray | list[list[list[float]]]) -> dict[str, Any]:
        """Predict one next session from exactly the previous configured sessions."""
        if self.artifacts is None:
            raise PredictionError("Fit or load the LSTM model before prediction.")
        values = np.asarray(session_windows, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, :, :]
        expected = (self.artifacts.sequence_length, len(self.artifacts.feature_columns))
        if values.ndim != 3 or values.shape[1:] != expected:
            raise PredictionError(
                f"Expected input shape (batch, {expected[0]}, {expected[1]}), got {values.shape}."
            )
        scaled = self.artifacts.scaler.transform(values.reshape(-1, values.shape[-1])).reshape(
            values.shape
        )
        self.artifacts.model.eval()
        with torch.no_grad():
            estimate, logits = self.artifacts.model(
                torch.tensor(scaled, dtype=torch.float32, device=self.device)
            )
        features = self.artifacts.scaler.inverse_transform(estimate.cpu().numpy())[0]
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {
            "predicted_category": self.artifacts.encoder.inverse_transform(
                [probabilities.argmax()]
            )[0],
            "category_confidence": float(probabilities.max()),
            "predicted_features": dict(
                zip(self.artifacts.feature_columns, features.tolist(), strict=True)
            ),
        }

    def save(self) -> Path:
        """Save the model and fitted preprocessing metadata in one Torch checkpoint."""
        if self.artifacts is None:
            raise LSTMTrainingError("No fitted model is available to save.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.artifacts.model
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": {
                    "input_size": len(self.artifacts.feature_columns),
                    "hidden_size": model.lstm.hidden_size,
                    "num_layers": model.lstm.num_layers,
                    "dropout": model.lstm.dropout,
                    "num_categories": len(self.artifacts.encoder.classes_),
                },
                "scaler_mean": self.artifacts.scaler.mean_,
                "scaler_scale": self.artifacts.scaler.scale_,
                "encoder_classes": self.artifacts.encoder.classes_,
                "feature_columns": self.artifacts.feature_columns,
                "sequence_length": self.artifacts.sequence_length,
            },
            self.model_path,
        )
        logger.info("LSTM model saved: %s", self.model_path)
        return self.model_path

    def load(self) -> LSTMArtifacts:
        """Load a checkpoint produced by :meth:`save`."""
        if not self.model_path.is_file():
            raise FileReadError(f"LSTM model not found: {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model = NextSessionLSTM(**checkpoint["model_config"]).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            scaler = StandardScaler()
            scaler.mean_ = checkpoint["scaler_mean"]
            scaler.scale_ = checkpoint["scaler_scale"]
            scaler.var_ = scaler.scale_**2
            scaler.n_features_in_ = len(scaler.mean_)
            encoder = LabelEncoder()
            encoder.classes_ = checkpoint["encoder_classes"]
        except (OSError, KeyError, RuntimeError) as exc:
            raise LSTMTrainingError(f"Unable to load LSTM checkpoint: {self.model_path}") from exc
        self.artifacts = LSTMArtifacts(
            model,
            scaler,
            encoder,
            list(checkpoint["feature_columns"]),
            int(checkpoint["sequence_length"]),
        )
        return self.artifacts

    def _fit(
        self, train: pd.DataFrame, validation: pd.DataFrame
    ) -> tuple[LSTMArtifacts, list[dict[str, float]]]:
        sequences, targets, categories, scaler, encoder = self._arrays(
            train, fit=True, include_processors=True
        )
        model = NextSessionLSTM(
            len(train.iloc[0]["target_features"]),
            int(self.settings["hidden_size"]),
            int(self.settings["num_layers"]),
            float(self.settings["dropout"]),
            len(encoder.classes_),
        ).to(self.device)
        self.artifacts = LSTMArtifacts(
            model,
            scaler,
            encoder,
            self._feature_columns(sequences.shape[-1]),
            sequences.shape[1],
        )
        val_sequences, val_targets, val_categories, _, _ = self._arrays(
            validation, fit=False, include_processors=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.settings["learning_rate"]))
        regression_loss, classification_loss = nn.MSELoss(), nn.CrossEntropyLoss()
        dataset = TensorDataset(
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(categories, dtype=torch.long),
        )
        loader = DataLoader(
            dataset, batch_size=min(int(self.settings["batch_size"]), len(dataset)), shuffle=False
        )
        best_state, best_loss, patience, history = None, float("inf"), 0, []
        weight = float(self.settings["classification_weight"])
        for epoch in range(int(self.settings["epochs"])):
            model.train()
            for batch_x, batch_y, batch_category in loader:
                optimizer.zero_grad()
                estimated, logits = model(batch_x.to(self.device))
                loss = regression_loss(
                    estimated, batch_y.to(self.device)
                ) + weight * classification_loss(logits, batch_category.to(self.device))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                estimated, logits = model(
                    torch.tensor(val_sequences, dtype=torch.float32, device=self.device)
                )
                val_loss = regression_loss(
                    estimated, torch.tensor(val_targets, dtype=torch.float32, device=self.device)
                ) + weight * classification_loss(
                    logits, torch.tensor(val_categories, dtype=torch.long, device=self.device)
                )
            value = float(val_loss.item())
            history.append({"epoch": epoch + 1, "validation_loss": value})
            if value < best_loss:
                best_loss, best_state, patience = (
                    value,
                    {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    0,
                )
            else:
                patience += 1
                if patience >= int(self.settings["early_stopping_patience"]):
                    break
        if best_state is None:
            raise LSTMTrainingError("LSTM training did not produce a model state.")
        model.load_state_dict(best_state)
        logger.info("LSTM trained | epochs=%d | best_validation_loss=%.4f", len(history), best_loss)
        return LSTMArtifacts(
            model, scaler, encoder, self._feature_columns(sequences.shape[-1]), sequences.shape[1]
        ), history

    def _arrays(self, frame: pd.DataFrame, fit: bool, include_processors: bool = False):
        sequences = np.stack(
            [np.vstack(value).astype(np.float64) for value in frame["feature_vectors"]]
        )
        targets = np.stack(
            [np.asarray(value, dtype=np.float64) for value in frame["target_features"]]
        )
        categories = frame["target_category"].astype(str).to_numpy()
        if not np.isfinite(sequences).all() or not np.isfinite(targets).all():
            raise DataValidationError("Sequence features must be finite numeric values.")
        if fit:
            scaler = StandardScaler().fit(
                np.vstack([sequences.reshape(-1, sequences.shape[-1]), targets])
            )
            encoder = LabelEncoder().fit(getattr(self, "_category_classes", categories))
        else:
            if self.artifacts is None:
                raise LSTMTrainingError("Preprocessors have not been fitted.")
            scaler, encoder = self.artifacts.scaler, self.artifacts.encoder
        sequences = scaler.transform(sequences.reshape(-1, sequences.shape[-1])).reshape(
            sequences.shape
        )
        targets = scaler.transform(targets)
        category_ids = encoder.transform(categories)
        return (
            (sequences, targets, category_ids, scaler, encoder)
            if include_processors
            else (sequences, targets, categories)
        )

    def _load_sequences(self) -> pd.DataFrame:
        if not self.input_path.is_file():
            raise FileReadError(f"Behavior sequences dataset not found: {self.input_path}")
        frame = pd.read_parquet(self.input_path)
        missing = self.REQUIRED_COLUMNS - set(frame.columns)
        if missing or len(frame) < 3:
            raise DataValidationError(
                f"Sequences need at least 3 rows and columns: {sorted(missing)}"
            )
        return frame.reset_index(drop=True)

    def _feature_columns(self, expected_count: int) -> list[str]:
        """Recover feature names from the Gold session dataset when available."""
        session_path = Path(get_paths()["paths"]["session_features_gold"]).expanduser()
        if session_path.is_file():
            try:
                frame = pd.read_parquet(session_path)
                _, columns = DatasetBuilder()._build_feature_matrix(frame)
                if len(columns) == expected_count:
                    return columns
            except (OSError, DataValidationError, KeyError):
                logger.warning("Could not recover LSTM feature names from %s", session_path)
        return [f"feature_{index}" for index in range(expected_count)]

    @staticmethod
    def _split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        train_end, validation_end = max(1, int(len(frame) * 0.8)), max(2, int(len(frame) * 0.9))
        return {
            "train": frame.iloc[:train_end],
            "validation": frame.iloc[train_end:validation_end],
            "test": frame.iloc[validation_end:],
        }

    def _save_predictions(self, predictions: pd.DataFrame) -> None:
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(self.predictions_path, index=False)

    def _log_mlflow(
        self, metrics: dict[str, float], history: list[dict[str, float]], samples: int
    ) -> str | None:
        if not self.track_mlflow:
            return None
        import mlflow
        import mlflow.pytorch

        setup_mlflow()
        experiment_id = get_experiment_id(get_experiment_name("lstm"))
        with mlflow.start_run(experiment_id=experiment_id, run_name="next-session-lstm") as run:
            mlflow.log_params(
                {key: value for key, value in self.settings.items() if not isinstance(value, dict)}
                | {"n_samples": samples, "device": str(self.device)}
            )
            mlflow.log_metrics(metrics)
            for entry in history:
                mlflow.log_metric(
                    "validation_loss", entry["validation_loss"], step=int(entry["epoch"])
                )
            if self.artifacts is not None and self.artifacts.model is not None:
                try:
                    mlflow.pytorch.log_model(
                        self.artifacts.model,
                        "lstm_model",
                        registered_model_name="NextSessionLSTMPredictor",
                        serialization_format="pickle",
                    )
                except Exception as err:
                    logger.warning("Could not register PyTorch model in MLflow registry: %s", err)

            mlflow.log_artifact(str(self.model_path))
            mlflow.log_artifact(str(self.predictions_path))
            return run.info.run_id

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
