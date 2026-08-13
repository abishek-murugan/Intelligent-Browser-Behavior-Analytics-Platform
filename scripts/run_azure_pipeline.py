"""
Azure ML entry point for the end-to-end analytics pipeline.

Prepares the raw inputs supplied by the Azure ML job (converting the RAM
usage CSV to the Parquet form the pipeline consumes when it is absent) and
then executes the same full pipeline used locally.

Local execution is unaffected: this script is only invoked on Azure.
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.ram_data_loader import RAMDataLoader
from src.pipeline import run_full_pipeline
from src.utils.config_loader import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_raw_inputs() -> None:
    """Ensure every raw input the pipeline expects exists under data/raw."""

    paths = get_paths()["paths"]

    parquet_path = Path(paths["ram_data_raw"]).expanduser()
    csv_path = Path(paths["ram_log"]).expanduser()

    if parquet_path.is_file():
        return

    if not csv_path.is_file():
        raise SystemExit(
            f"Neither {parquet_path} nor {csv_path} was found under data/raw."
        )

    loader = RAMDataLoader()
    logger.info("Converting RAM usage CSV to Parquet: %s", csv_path)
    loader.save(loader.load())


if __name__ == "__main__":
    prepare_raw_inputs()
    run_full_pipeline()
