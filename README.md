# Time-Based Browsing Pattern Analyzer

### Deep learning analysis of time-based browsing patterns with RAM usage correlation.

Modern web users generate thousands of browsing events every day. Organizations can use this behavioral data to understand productivity, browsing habits, predict future user behavior, detect anomalies, and generate personalized recommendations. This project builds an end-to-end analytics platform that processes browser history and system usage logs using Databricks and PySpark, applies machine learning and deep learning models to extract insights, and presents them through an interactive dashboard.

### Domain

Productivity & Digital Behavior Analytics Platform

## Completed modelling stages

The project now includes the following production-oriented workflow:

1. Session clustering with KMeans and MLflow experiment tracking.
2. A PyTorch multi-task LSTM that uses the previous **five browsing sessions**
   to forecast the next session's numeric profile and dominant category.
3. A transparent recommendation engine that puts the LSTM category forecast
   first, then ranks historically frequent categories.
4. A Streamlit dashboard that displays session segmentation, forecast quality,
   category forecasts, and per-session recommendations.

Each stage emits structured logs and tracks its metrics/artifacts in MLflow.

## Run the workflow

Run these commands from the repository root after installing the project
dependencies with `uv sync`:

```bash
# Rebuild Gold data using five-session windows, train the model, and log MLflow artifacts.
uv run python -c "from src.modeling.dataset_builder import DatasetBuilder; from src.modeling.lstm_pipeline import LSTMPipeline; DatasetBuilder(sequence_length=5).run(); LSTMPipeline().run()"

# Produce prediction-aware category recommendations.
uv run python -c "from src.recommendation.pipeline import RecommendationPipeline; RecommendationPipeline().run()"

# Launch the dashboard.
uv run streamlit run app.py
```

The same workflows are available as dedicated notebooks:

- `notebooks/03_lstm_next_session.ipynb`
- `notebooks/04_recommendations.ipynb`

Artifacts are written under `data/gold/`, `models/`, and MLflow's configured
tracking URI (`mlruns` locally by default). Set `MLFLOW_TRACKING_URI` to use a
remote MLflow server.
