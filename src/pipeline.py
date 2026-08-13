"""
End-to-end pipeline orchestration.

Ties together every stage from raw inputs through to recommendations.
Single source of truth shared by the CLI, the Streamlit app and CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.clustering.pipeline import ClusteringPipeline
from src.deep_learning.dataset_builder import DatasetBuilder
from src.deep_learning.lstm_pipeline import LSTMPipeline
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.preprocessing.data_integrator import BrowserRAMIntegrator
from src.preprocessing.domain_mapper import DomainMapper
from src.preprocessing.sessionizer import Sessionizer
from src.recommendation.pipeline import RecommendationPipeline
from src.utils.config_loader import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _integrate() -> None:
    """Integrate and persist the aligned browser/RAM dataset."""
    integrator = BrowserRAMIntegrator()
    integrator.save(integrator.integrate())


def _categorize() -> None:
    """Categorize domains and persist the categorized dataset."""
    paths = get_paths()["paths"]
    aligned = pd.read_parquet(paths["browser_ram_aligned"])
    DomainMapper(mapping_path=paths["domain_category_map"]).run(aligned)


def _publish_gold_features() -> None:
    """Publish the silver session features to the gold layer."""
    paths = get_paths()["paths"]
    features = pd.read_parquet(paths["session_features"])
    gold_path = Path(paths["session_features_gold"])
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(gold_path, index=False)
    logger.info("Gold session features published: %s | sessions=%d", gold_path, len(features))


def run_full_pipeline() -> None:
    """Execute every stage in dependency order."""
    logger.info("Stage 1/9: integrating browser and RAM data")
    _integrate()

    logger.info("Stage 2/9: mapping domains to categories")
    _categorize()

    logger.info("Stage 3/9: sessionizing browsing events")
    Sessionizer().run()

    logger.info("Stage 4/9: engineering session features")
    FeaturePipeline().run()

    logger.info("Stage 5/9: publishing gold session features")
    _publish_gold_features()

    logger.info("Stage 6/9: clustering sessions")
    ClusteringPipeline().run()

    logger.info("Stage 7/9: building next-session sequences")
    DatasetBuilder(sequence_length=5).run()

    logger.info("Stage 8/9: training LSTM and forecasting next categories")
    LSTMPipeline().run()

    logger.info("Stage 9/9: generating recommendations")
    RecommendationPipeline().run()

    logger.info("Full pipeline completed")
