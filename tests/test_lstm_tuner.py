import numpy as np
import optuna
import pandas as pd
import pytest

from src.deep_learning.lstm_tuner import LSTMTuner
from src.exceptions import LSTMTuningError


def _sequences(count: int = 60) -> pd.DataFrame:
    rows = []
    for index in range(count):
        category = "Work" if index % 2 else "Learning"
        rows.append(
            {
                "target_session_id": index + 6,
                "target_category": category,
                "feature_vectors": [[index, index + 1], [index + 1, index + 2]],
                "target_features": [index + 2, index + 3],
            }
        )
    return pd.DataFrame(rows)


SEARCH_SPACE = {
    "hidden_size": {"min": 8, "max": 16},
    "num_layers": {"min": 1, "max": 2},
    "dropout": {"min": 0.0, "max": 0.1},
    "learning_rate": {"min": 0.001, "max": 0.01},
    "batch_size": {"min": 16, "max": 32},
    "classification_weight": {"min": 0.5, "max": 0.5},
}


def test_tuner_optimizes_and_returns_best_params(tmp_path):
    sequences = _sequences()
    input_path = tmp_path / "sequences.parquet"
    sequences.to_parquet(input_path, index=False)

    tuner = LSTMTuner(
        input_path=input_path,
        track_mlflow=False,
        device="cpu",
        n_trials=2,
        search_space=SEARCH_SPACE,
        study_name="test-tuning",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    result = tuner.run()

    assert result.n_trials == 2
    assert result.best_value >= 0.0
    assert np.isfinite(result.best_value)
    assert set(result.best_params) == set(SEARCH_SPACE)
    assert isinstance(result.study, optuna.study.Study)


def test_tuner_reuses_existing_study(tmp_path):
    sequences = _sequences()
    input_path = tmp_path / "sequences.parquet"
    sequences.to_parquet(input_path, index=False)

    kwargs = {
        "input_path": input_path,
        "track_mlflow": False,
        "device": "cpu",
        "n_trials": 1,
        "search_space": SEARCH_SPACE,
        "study_name": "reused-study",
        "storage": f"sqlite:///{tmp_path / 'reused.db'}",
    }
    first = LSTMTuner(**kwargs).run()
    second = LSTMTuner(**kwargs).run()

    assert second.n_trials == 2
    assert first.study.study_name == second.study.study_name


def test_tuner_rejects_empty_search_space(tmp_path):
    sequences = _sequences()
    input_path = tmp_path / "sequences.parquet"
    sequences.to_parquet(input_path, index=False)

    tuner = LSTMTuner(
        input_path=input_path,
        track_mlflow=False,
        search_space={},
        study_name="empty-space",
        storage=f"sqlite:///{tmp_path / 'empty.db'}",
    )
    tuner._prepare_data()

    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    with pytest.raises(LSTMTuningError, match="search space"):
        tuner._suggest_params(trial)


def test_tuner_missing_input_raises(tmp_path):
    tuner = LSTMTuner(
        input_path=tmp_path / "missing.parquet",
        track_mlflow=False,
    )
    with pytest.raises(LSTMTuningError, match="not found"):
        tuner._prepare_data()
