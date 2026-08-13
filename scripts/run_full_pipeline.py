"""
Run the full end-to-end analytics pipeline from raw inputs to recommendations.

Chain: integration -> categorization -> sessionization -> feature engineering
-> gold publish -> clustering -> LSTM sequence builder -> LSTM
training/prediction -> recommendations.

Usage:
    uv run python scripts/run_full_pipeline.py
"""

from __future__ import annotations

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
