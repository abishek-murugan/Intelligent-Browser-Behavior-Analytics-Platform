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
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow server --backend-store-uri mlruns --host 127.0.0.1 --port 5000
# equivalent shorthand:
# MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri mlruns --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

> **Note:** MLflow 3.x keeps the local filesystem backend (`mlruns`) in maintenance mode and
> refuses to start the server against it unless `MLFLOW_ALLOW_FILE_STORE=true` is set in the
> shell environment. The project's `.env` is **not** auto-loaded, so the flag must be exported
> in the shell (or via `set -a; . ./.env; set +a`). The same flag already gates client-side
> logging in `src/utils/mlflow_utils.py`.
>
> If you later outgrow the file store, the supported path is to migrate to a database backend:
> `mlflow migrate-filestore --backend-store-uri mlruns --tracking-uri sqlite:///mlruns/mlflow.db`,
> then point `--backend-store-uri` at `sqlite:///mlruns/mlflow.db`.

---

## 🛠️ Quickstart & Environment Setup

This project uses `uv` for fast, reproducible environment and package management.

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Complete End-to-End Pipeline
```bash
# Requires raw inputs under data/raw (Chrome history, RAM usage, domain map).
# Optional: generate a deterministic synthetic dataset first (no real profile needed):
uv run python scripts/make_synthetic_data.py

# Full chain: integration -> categorization -> sessionization -> feature
# engineering -> clustering -> LSTM training/prediction -> recommendations.
uv run python scripts/run_full_pipeline.py
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
├── scripts/                     # Synthetic data generation & full-pipeline CLI
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

---

## ☁️ Azure ML Deployment

The production path runs the same pipeline on Azure ML from real raw data
stored in Azure Blob Storage. A **managed environment** (curated base image +
conda file, built by Azure ML in the workspace container registry) provides a
reproducible runtime — **no local Docker and no Dockerfile**. The job code is
pushed to GitHub and referenced as the job's code source.

### Azure resources

Already provisioned and reused (no new resources are created):

| Resource | Name | Purpose |
| --- | --- | --- |
| Workspace | `mlw-browser-analytics` | Azure ML workspace |
| Resource group | `rg-browser-analytics` | Resource grouping |
| Compute cluster | `browser-analytics-cluster` | CPU job compute |
| Storage account | `browseranalyticsabishek` | Blob storage |
| Blob containers | `raw`, `silver`, `gold` | Raw inputs / outputs |
| Datastore | `browser_analytics_storage` | Points at the `raw` container |
| Container registry | workspace default ACR | Hosts the built environment image |
| GitHub repo | `abishek-murugan/Intelligent-Browser-Behavior-Analytics-Platform` | Source-of-truth for the code (`feature/azure`) |

The workspace's default Azure Container Registry is used automatically by
managed environment builds — no extra ACR resource or credential is required.

### Push the code

The job code source is the local project checkout (`code: ..`), which Azure ML
uploads to the workspace and automatically records the git repository, branch,
and commit as job properties. Push the code first so the commit is the source
of truth:

```bash
git add -A
git commit -m "Add Azure ML deployment (managed environment)"
git push origin feature/azure
```

The branch pushed is `feature/azure`, referenced by `azure/job.yml`.

### Raw data on Azure Storage

Upload the three raw inputs to the `raw` container (the datastore root):

```bash
az storage blob upload --container-name raw --account-name browseranalyticsabishek \
  --name chrome_history.parquet --file data/raw/chrome_history.parquet --auth-mode login
az storage blob upload --container-name raw --account-name browseranalyticsabishek \
  --name ram_usage.csv --file data/raw/ram_usage.csv --auth-mode login
az storage blob upload --container-name raw --account-name browseranalyticsabishek \
  --name domain_category_map.csv --file data/raw/domain_category_map.csv --auth-mode login
```

The job downloads these as the `raw_data` input into `data/raw/`. The Azure
entry point `scripts/run_azure_pipeline.py` converts `ram_usage.csv` to
`ram_usage.parquet` (the form the pipeline expects) when the Parquet is absent,
then runs the identical local pipeline — Azure **executes** the pipeline and
generates silver/gold/models rather than receiving pre-generated outputs.

### Build and register the environment

Azure ML builds the managed environment from the curated base image +
`azure/conda-dependencies.yml` in the workspace ACR (no local Docker):

```bash
az ml environment create -f azure/env.yml -g rg-browser-analytics -w mlw-browser-analytics
```

This registers a versioned `browser-analytics-env` environment. Confirm the
version:

```bash
az ml environment show -n browser-analytics-env -g rg-browser-analytics -w mlw-browser-analytics
```

If the version is not `1` (e.g. after a rebuild), update the version reference
in `azure/job.yml`.

### Submit the Azure ML job

```bash
# Validate the job spec without submitting (dry run):
az ml job create -f azure/job.yml -g rg-browser-analytics -w mlw-browser-analytics --dry-run

# Submit:
az ml job create -f azure/job.yml -g rg-browser-analytics -w mlw-browser-analytics
```

The job:

1. uploads the local project checkout (`code: ..`) and records the git
   repo/branch/commit,
2. downloads `raw_data` from `browser_analytics_storage` into `data/raw/`,
3. runs `scripts/run_azure_pipeline.py` in the `browser-analytics-env` managed
   environment,
4. uploads `data/silver/`, `data/gold/` and `models/` to the workspace default
   datastore as job outputs (`silver`, `gold`, `models`).

### Monitor the job

```bash
# Live logs:
az ml job stream -n <job-name> -g rg-browser-analytics -w mlw-browser-analytics

# Status / metadata:
az ml job show -n <job-name> -g rg-browser-analytics -w mlw-browser-analytics

# Download job outputs after completion:
az ml job download -n <job-name> -g rg-browser-analytics -w mlw-browser-analytics \
  --download-type all
```

### MLflow on Azure

Azure ML command jobs inject the `MLFLOW_TRACKING_URI`
(`azureml://<region>.api.azureml.ms/mlflow/v1.0/...`) and authentication
environment variables automatically. `src/utils/mlflow_utils.py` already
resolves `MLFLOW_TRACKING_URI` from the environment first, so the existing
experiments (`browser-behavior-clustering`, `browser-behavior-lstm`,
`browser-behavior-lstm-tuning`, `browser-behavior-recommendation`), runs,
metrics, and model artifacts are tracked in the Azure ML workspace with **no
code changes**.

Local execution is unchanged: with no `MLFLOW_TRACKING_URI` set, runs still
land in the local `mlruns/` directory.

---

## 🗺️ Architecture

```
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Azure Blob: raw  │──▶│ Azure ML command job │──▶│ project checkout     │
│ chrome_history   │   │ browser-analytics-   │   │ (code: .., git-tracked│
│ ram_usage.csv    │   │ cluster (CPU)        │   │  branch/commit)       │
│ domain_map.csv   │   └──────────┬───────────┘   └──────────┬───────────┘
└──────────────────┘              │                          │
                                  │ data/raw/                 │
                                  ▼                           ▼
                       scripts/run_azure_pipeline.py ──► src/pipeline.run_full_pipeline()
                                  │
             ┌────────────────────┼─────────────────────────┐
             ▼                    ▼                         ▼
      data/silver/*         data/gold/*               models/*
             │                    │                         │
             ▼                    ▼                         │
      Azure ML MLflow (runs, metrics, models, artifacts)   │
             │                                              │
             └────────────► job outputs (default datastore) ┘
```

The Azure job mirrors the local flow exactly: integration → domain
categorization → sessionization → feature engineering → gold publish →
clustering → sequence building → LSTM training/prediction → recommendations.
Raw inputs come from Azure Storage, results and MLflow artifacts are persisted
to Azure, and the same pipeline can be reproduced locally on synthetic data.
