"""
Domain category mapping for browser behavior analysis.

Maps browser domains to the categories defined in
domain_category_map.csv.

The mapper uses:
1. Normalized exact matching
2. Parent-domain matching
3. Uncategorized fallback

This preserves explicit subdomain mappings when they exist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exceptions import (
    DataValidationError,
    FileReadError,
    FileWriteError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DomainMapper:
    """Map browser domains to predefined behavioral categories."""

    REQUIRED_DATA_COLUMNS = {
        "timestamp",
        "url",
        "title",
        "domain",
        "visit_count",
        "total_mb",
        "used_mb",
        "available_mb",
        "usage_percent",
    }

    REQUIRED_MAPPING_COLUMNS = {
        "domain",
        "category",
    }

    def __init__(
        self,
        mapping_path: str | Path,
        output_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the domain mapper.

        Parameters
        ----------
        mapping_path:
            Path to domain_category_map.csv.

        output_path:
            Path where the categorized Parquet dataset will be saved.
        """

        self.mapping_path = Path(mapping_path).expanduser()

        self.output_path = Path(
            output_path
            if output_path is not None
            else "data/silver/browser_ram_categorized.parquet"
        ).expanduser()

    @staticmethod
    def normalize_domain(domain: object) -> str | None:
        """
        Normalize a domain for matching.

        Examples
        --------
        www.instagram.com
            -> instagram.com

        https://www.instagram.com/
            -> instagram.com

        instagram.com/
            -> instagram.com
        """

        if pd.isna(domain):
            return None

        domain = str(domain).strip().lower()

        if not domain:
            return None

        # Remove protocol if present.
        if domain.startswith("https://"):
            domain = domain[8:]

        elif domain.startswith("http://"):
            domain = domain[7:]

        # Remove trailing slash.
        domain = domain.rstrip("/")

        # Remove port from normal hostnames.
        if ":" in domain:
            host, port = domain.rsplit(":", 1)

            if port.isdigit():
                domain = host

        # Remove www prefix.
        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    def load_mapping(self) -> pd.DataFrame:
        """
        Load and validate the domain-category mapping.

        Returns
        -------
        pd.DataFrame
            Validated domain-category mapping.
        """

        if not self.mapping_path.is_file():
            raise FileReadError(f"Domain mapping file not found: {self.mapping_path}")

        try:
            mapping = pd.read_csv(
                self.mapping_path,
            )

        except (
            OSError,
            pd.errors.ParserError,
        ) as exc:
            raise FileReadError(f"Failed to read domain mapping file: {self.mapping_path}") from exc

        missing_columns = self.REQUIRED_MAPPING_COLUMNS - set(mapping.columns)

        if missing_columns:
            raise DataValidationError(
                f"Domain mapping is missing required columns: {sorted(missing_columns)}"
            )

        if mapping.empty:
            raise DataValidationError("Domain mapping file is empty.")

        mapping = mapping[["domain", "category"]].copy()

        mapping["domain"] = mapping["domain"].apply(self.normalize_domain)

        mapping["category"] = mapping["category"].astype("string").str.strip()

        mapping = mapping.dropna(
            subset=[
                "domain",
                "category",
            ]
        )

        # Check for duplicate normalized domains.
        duplicate_domains = mapping[
            mapping["domain"].duplicated(
                keep=False,
            )
        ]

        if not duplicate_domains.empty:
            duplicates = duplicate_domains["domain"].unique().tolist()

            raise DataValidationError(
                f"Domain mapping contains duplicate normalized domains: {duplicates[:20]}"
            )

        logger.info(
            "Domain mapping loaded | records=%d | categories=%d",
            len(mapping),
            mapping["category"].nunique(),
        )

        return mapping

    def map_domains(
        self,
        dataframe: pd.DataFrame,
        mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add behavioral categories to browser events.

        Matching strategy
        ------------------
        1. Normalize browser domain.
        2. Try exact normalized domain match.
        3. If no exact match exists, try parent domains.
        4. Assign 'Uncategorized' if no match exists.

        Explicit subdomain mappings always take priority.

        For example:

            studio.youtube.com
                -> studio.youtube.com mapping

        If that does not exist:

            something.example.com
                -> example.com mapping
        """

        # --------------------------------------------------
        # Validate browser dataset
        # --------------------------------------------------

        missing_columns = self.REQUIRED_DATA_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Browser dataset is missing required columns: {sorted(missing_columns)}"
            )

        # --------------------------------------------------
        # Validate mapping dataset
        # --------------------------------------------------

        missing_mapping_columns = self.REQUIRED_MAPPING_COLUMNS - set(mapping.columns)

        if missing_mapping_columns:
            raise DataValidationError(
                f"Domain mapping is missing required columns: {sorted(missing_mapping_columns)}"
            )

        result = dataframe.copy()
        mapping = mapping.copy()

        # --------------------------------------------------
        # Normalize mapping domains
        # --------------------------------------------------

        mapping["domain_normalized"] = mapping["domain"].apply(self.normalize_domain)

        # Create lookup dictionary.
        domain_to_category = dict(
            zip(
                mapping["domain_normalized"],
                mapping["category"],
            )
        )

        mapping_domains = set(mapping["domain_normalized"].dropna())

        # --------------------------------------------------
        # Normalize browser domains
        # --------------------------------------------------

        result["domain_normalized"] = result["domain"].apply(self.normalize_domain)

        # --------------------------------------------------
        # First pass:
        # Exact normalized domain matching
        # --------------------------------------------------

        result["category"] = result["domain_normalized"].map(domain_to_category)

        exact_match_count = int(result["category"].notna().sum())

        logger.info(
            "Exact domain matches: %d",
            exact_match_count,
        )

        # --------------------------------------------------
        # Second pass:
        # Parent-domain matching
        # --------------------------------------------------

        unmatched_indices = result.index[result["category"].isna()]

        parent_match_count = 0

        for index in unmatched_indices:
            domain = result.at[
                index,
                "domain_normalized",
            ]

            if not domain or not isinstance(
                domain,
                str,
            ):
                continue

            parts = domain.split(".")

            # Need at least:
            # example.com
            #
            # Therefore a domain with fewer than
            # two components cannot have a parent
            # domain.
            if len(parts) < 2:
                continue

            # Try progressively broader domains.
            #
            # Example:
            #
            # a.b.example.com
            #
            # b.example.com
            # example.com
            #
            # The closest matching parent is used.
            for i in range(1, len(parts) - 1):
                parent_domain = ".".join(parts[i:])

                if parent_domain in mapping_domains:
                    result.at[
                        index,
                        "category",
                    ] = domain_to_category[parent_domain]

                    parent_match_count += 1

                    break

        logger.info(
            "Parent-domain matches: %d",
            parent_match_count,
        )

        # --------------------------------------------------
        # Final fallback
        # --------------------------------------------------

        result["category"] = result["category"].fillna("Uncategorized").astype("string")

        # Remove temporary column.
        result = result.drop(
            columns=["domain_normalized"],
        )

        # --------------------------------------------------
        # Mapping statistics
        # --------------------------------------------------

        total_records = len(result)

        mapped_records = int((result["category"] != "Uncategorized").sum())

        unmapped_records = int((result["category"] == "Uncategorized").sum())

        coverage = mapped_records / total_records * 100 if total_records > 0 else 0.0

        logger.info(
            "Domain mapping completed | total=%d | mapped=%d | unmapped=%d | coverage=%.2f%%",
            total_records,
            mapped_records,
            unmapped_records,
            coverage,
        )

        return result

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the categorized dataset as Parquet.

        Parameters
        ----------
        dataframe:
            Categorized browser dataset.

        output_path:
            Optional output path.

        Returns
        -------
        Path
            Path to the saved dataset.
        """

        path = Path(output_path if output_path is not None else self.output_path).expanduser()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            dataframe.to_parquet(
                path,
                index=False,
            )

        except (
            OSError,
            ImportError,
        ) as exc:
            raise FileWriteError(f"Failed to save categorized dataset: {path}") from exc

        logger.info(
            "Categorized dataset saved: %s | records=%d",
            path,
            len(dataframe),
        )

        return path

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the complete domain mapping pipeline.

        Parameters
        ----------
        dataframe:
            Integrated browser/RAM dataset.

        Returns
        -------
        pd.DataFrame
            Categorized dataset.
        """

        logger.info("Starting domain categorization.")

        mapping = self.load_mapping()

        result = self.map_domains(
            dataframe,
            mapping,
        )

        self.save(result)

        logger.info("Domain categorization completed.")

        return result
