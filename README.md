# Time-Based Browsing Pattern Analyzer

### Deep learning analysis of time-based user browsing patterns with system RAM correlation.

Modern web users generate thousands of browsing events every day. Organizations and power users can utilize this behavioral data to understand productivity patterns, optimize system memory allocation, predict future user behavior, detect workflow fragmentation, and deliver personalized contextual recommendations. 

This repository provides an end-to-end production ML/DL analytics platform that processes browser history and system memory telemetry logs, applies unsupervised **KMeans clustering** and deep **PyTorch LSTM multi-task sequence models**, tracks all experiments in **MLflow**, and serves an interactive **Streamlit dashboard**.

---

## 🚀 Domain & System Architecture

```
                                 ┌────────────────────────┐
                                 │ Chrome Browsing Logs   │
                                 └───────────┬────────────┘
                                             │
                                             ▼
 ┌────────────────────────┐      ┌────────────────────────┐
 │ System RAM Telemetry   ├─────►│ Data Integrator        │
 └────────────────────────┘      └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │ Feature Engineering    │
                                 └───────────┬────────────┘
                                             │
                  ┌──────────────────────────┴──────────────────────────┐
                  ▼                                                     ▼
     ┌────────────────────────┐                            ┌────────────────────────┐
     │ KMeans Clustering      │                            │ PyTorch Multi-Task LSTM│
     └───────────┬────────────┘                            └───────────┬────────────┘
                 │                                                     │
                 └──────────────────────────┬──────────────────────────┘
                                            ▼
                                ┌────────────────────────┐
                                │ Recommendation Engine  │
                                └───────────┬────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Streamlit Dashboard    │
                                └────────────────────────┘
```

---

## ✨ Key Features

1. **Integrated Data Ingestion**: Synchronizes Chrome history with 5-second interval RAM telemetry.
2. **KMeans Session Clustering**: Segments browsing sessions based on duration, switching rate, page volume, and RAM consumption profiles.
3. **PyTorch Multi-Task LSTM**: Uses sliding 5-session windows to forecast:
   - Next-session dominant category (classification).
   - Continuous next-session duration and memory footprint (regression).
4. **Contextual Recommendation Engine**: Combines model predictions with rule-based heuristics to yield severity-ranked actionable advice (🔴 High, 🟠 Medium, 🟢 Low).
5. **Centralized MLflow Tracking**: Logs all runs, metrics, parameters, loss curves, and model state dicts to an absolute local store.
6. **Interactive Streamlit Dashboard**: Provides real-time PCA cluster visualizers, RAM time series, and an interactive next-category prediction sandbox.

---

## 📊 MLflow Tracking Server

All pipelines log tracking data centrally to `<PROJECT_ROOT>/mlruns`. Relative path resolution is configured to eliminate fragmented tracking folders.

### Launch MLflow UI
To inspect training loss curves, hyperparameter runs, and saved model artifacts:

```bash
uv run mlflow ui --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🛠️ Quickstart & Environment Setup

This project uses `uv` for fast, reproducible environment and package management.

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Complete End-to-End Pipeline
```bash
# Execute Feature Engineering & KMeans Clustering
uv run python -c "from src.feature_engineering.feature_pipeline import FeaturePipeline; from src.clustering.pipeline import ClusteringPipeline; FeaturePipeline().run(); ClusteringPipeline().run()"

# Build 5-Session Gold Sequences, Train PyTorch LSTM, and Generate Recommendations
uv run python -c "from src.deep_learning.dataset_builder import DatasetBuilder; from src.deep_learning.lstm_pipeline import LSTMPipeline; from src.recommendation.pipeline import RecommendationPipeline; DatasetBuilder(sequence_length=5).run(); LSTMPipeline().run(); RecommendationPipeline().run()"
```

### 3. Launch Streamlit Analytics Dashboard
```bash
uv run streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) to view the interactive dashboard.

---

## 📓 Jupyter Notebooks

Interactive, educational notebooks for experimentation and analysis are located under `notebooks/`:

- `notebooks/01_eda_browser_ram.ipynb`: Exploratory Data Analysis & RAM Correlation.
- `notebooks/02_clustering.ipynb`: Unsupervised KMeans Session Segmentation & PCA.
- `notebooks/03_lstm_next_session.ipynb`: Deep Learning PyTorch Multi-Task LSTM Training & Evaluation.
- `notebooks/04_recommendations.ipynb`: Prediction-Driven Recommendation Engine & Severity Ranking.

---

## 📁 Project Directory Structure

```
├── README.md
├── app.py                       # Interactive Streamlit Dashboard
├── config/                      # YAML configuration files
├── data/                        # Processed Parquet artifacts (raw, bronze, silver, gold)
├── logs/                        # System execution log files
├── mlruns/                      # Centralized MLflow experiment store
├── models/                      # PyTorch and Scikit-Learn saved models
├── notebooks/                   # Step-by-step experiment notebooks
├── pyproject.toml               # Project metadata & dependencies
├── reports/                     # Markdown project reports & figures
│   ├── Final_Project_Report.md
│   ├── clustering/
│   ├── lstm/
│   ├── recommendation/
│   └── images/                  # Generated chart images
├── src/                         # Production Python source modules
│   ├── clustering/
│   ├── deep_learning/
│   ├── feature_engineering/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── recommendation/
│   └── utils/
└── tests/                       # Pytest automated test suite
```

---

## 🧪 Testing & Verification

Run the full pytest suite:

```bash
uv run pytest
```
