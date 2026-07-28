# Time-Based Browsing Pattern Analyzer

Deep learning analysis of time-based browsing patterns with RAM usage correlation.

## Project Structure

```
├── config/            YAML configuration files
├── src/
│   ├── collector/     Data collection (RAM, browser history)
│   ├── dashboard/     Visualization dashboard
│   ├── models/        Deep learning models (LSTM, autoencoder)
│   ├── preprocessing/ Data pipeline and feature engineering
│   ├── recommendation/ Recommendation engine
│   └── utils/         Shared utilities (config loader, etc.)
├── tests/             Test suite
└── pyproject.toml     Project metadata and dependencies
```

## Quick Start

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## Configuration

All configuration lives in `config/` as YAML files:

| File              | Purpose                        |
|-------------------|--------------------------------|
| `config.yaml`     | General project settings       |
| `paths.yaml`      | Data and artifact paths        |
| `models.yaml`     | Model hyperparameters          |
| `logging.yaml`    | Logging configuration          |
| `azure.yaml`      | Azure/Azure ML settings        |
| `dashboard.yaml`  | Dashboard refresh and defaults |

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format .
```
