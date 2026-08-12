from pathlib import Path

import pandas as pd
import pytest

from src.exceptions import DataValidationError, FileReadError
from src.preprocessing.domain_mapper import DomainMapper


def _write_mapping_csv(
    path: Path,
    rows: list[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    mapping = pd.DataFrame(rows, columns=["domain", "category"])

    mapping.to_csv(path, index=False)


@pytest.fixture
def mapping_csv(tmp_path) -> Path:
    path = tmp_path / "domain_category_map.csv"

    _write_mapping_csv(
        path,
        [
            ("google.com", "Search/Reference"),
            ("github.com", "Development/Programming"),
            ("instagram.com", "Social Media"),
            ("stackoverflow.com", "Development/Programming"),
            ("youtube.com", "Video Streaming"),
            ("studio.youtube.com", "Content Management"),
            ("example.com", "Reference"),
        ],
    )

    return path


@pytest.fixture
def mapper(mapping_csv, tmp_path) -> DomainMapper:
    return DomainMapper(
        mapping_path=mapping_csv,
        output_path=tmp_path / "browser_ram_categorized.parquet",
    )


def test_normalize_domain():
    normalize = DomainMapper.normalize_domain

    assert normalize("www.instagram.com") == "instagram.com"
    assert normalize("https://www.instagram.com/") == "instagram.com"
    assert normalize("instagram.com/") == "instagram.com"
    assert normalize("http://example.com:8080") == "example.com"
    assert normalize("  GITHUB.COM  ") == "github.com"
    assert normalize("") is None
    assert normalize(pd.NA) is None
    assert normalize(pd.NaT) is None


def test_load_mapping(mapper, mapping_csv):
    mapping = mapper.load_mapping()

    assert set(mapping.columns) == {"domain", "category"}
    assert len(mapping) == 7

    assert "google.com" in mapping["domain"].values
    assert (
        mapping.loc[
            mapping["domain"] == "studio.youtube.com",
            "category",
        ].iloc[0]
        == "Content Management"
    )


def test_load_mapping_missing_file_raises(tmp_path):
    with pytest.raises(FileReadError, match="not found"):
        DomainMapper(tmp_path / "missing.csv").load_mapping()


def test_load_mapping_missing_columns_raises(tmp_path):
    path = tmp_path / "map.csv"

    _write_mapping_csv(path, [("google.com", "Search")])

    path.write_text("domain\ngoogle.com\n")

    with pytest.raises(DataValidationError, match="category"):
        DomainMapper(path).load_mapping()


def test_load_mapping_empty_raises(tmp_path):
    path = tmp_path / "map.csv"

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["domain", "category"]).to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="empty"):
        DomainMapper(path).load_mapping()


def test_load_mapping_duplicate_domains_raise(tmp_path):
    path = tmp_path / "map.csv"

    _write_mapping_csv(
        path,
        [
            ("google.com", "Search"),
            ("www.google.com", "Other"),
        ],
    )

    with pytest.raises(DataValidationError, match="duplicate"):
        DomainMapper(path).load_mapping()


def test_map_domains_exact_match(mapper, integrated_frame):
    result = mapper.map_domains(
        integrated_frame(),
        mapper.load_mapping(),
    )

    categories = (
        result[["domain", "category"]].drop_duplicates("domain").set_index("domain")["category"]
    )

    assert categories["www.google.com"] == "Search/Reference"
    assert categories["github.com"] == "Development/Programming"
    assert categories["www.instagram.com"] == "Social Media"
    assert categories["stackoverflow.com"] == "Development/Programming"


def test_map_domains_subdomain_fallback(mapper, integrated_frame):
    mapping = mapper.load_mapping()

    frame = integrated_frame()
    frame["domain"] = "sub.example.com"

    result = mapper.map_domains(frame, mapping)

    assert (result["category"] == "Reference").all()


def test_map_domains_explicit_subdomain_priority(mapper, integrated_frame):
    mapping = mapper.load_mapping()

    frame = integrated_frame()
    frame["domain"] = "studio.youtube.com"

    result = mapper.map_domains(frame, mapping)

    assert (result["category"] == "Content Management").all()


def test_map_domains_uncategorized_fallback(mapper, integrated_frame):
    mapping = mapper.load_mapping()

    frame = integrated_frame()
    frame["domain"] = "unknown-site.dev"

    result = mapper.map_domains(frame, mapping)

    assert (result["category"] == "Uncategorized").all()


def test_map_domains_missing_columns_raise(mapper, integrated_frame):
    mapping = mapper.load_mapping()

    frame = integrated_frame().drop(columns=["visit_count"])

    with pytest.raises(DataValidationError, match="visit_count"):
        mapper.map_domains(frame, mapping)


def test_save_writes_parquet(mapper, integrated_frame):
    mapping = mapper.load_mapping()

    result = mapper.map_domains(integrated_frame(), mapping)

    path = mapper.save(result)

    assert path.exists()

    loaded = pd.read_parquet(path)

    pd.testing.assert_frame_equal(loaded, result)


def test_run_end_to_end(mapper, integrated_frame, tmp_path):
    result = mapper.run(integrated_frame())

    assert len(result) == 6

    output = tmp_path / "browser_ram_categorized.parquet"

    assert output.exists()
    assert (result["category"] != "Uncategorized").all()
