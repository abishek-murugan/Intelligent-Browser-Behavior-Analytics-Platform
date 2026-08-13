import numpy as np
import pandas as pd

from src.deep_learning.lstm_pipeline import LSTMPipeline


def _sequences(count: int = 30) -> pd.DataFrame:
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


def test_lstm_pipeline_trains_saves_loads_and_predicts(tmp_path):
    sequences = _sequences()
    input_path = tmp_path / "sequences.parquet"
    model_path = tmp_path / "model.pt"
    prediction_path = tmp_path / "predictions.parquet"
    sequences.to_parquet(input_path, index=False)

    pipeline = LSTMPipeline(
        input_path=input_path,
        model_path=model_path,
        predictions_path=prediction_path,
        track_mlflow=False,
        device="cpu",
        overrides={
            "hidden_size": 8,
            "num_layers": 1,
            "epochs": 3,
            "batch_size": 8,
            "early_stopping_patience": 2,
        },
    )
    result = pipeline.run()

    assert model_path.is_file()
    assert prediction_path.is_file()
    assert {"rmse", "mae", "category_accuracy"} <= result["metrics"].keys()
    assert len(result["predictions"]) == 3

    restored = LSTMPipeline(model_path=model_path, track_mlflow=False, device="cpu")
    restored.load()
    prediction = restored.predict_next(np.asarray(sequences.iloc[0]["feature_vectors"]))
    assert prediction["predicted_category"] in {"Work", "Learning"}
    assert set(prediction["predicted_features"]) == {"feature_0", "feature_1"}
