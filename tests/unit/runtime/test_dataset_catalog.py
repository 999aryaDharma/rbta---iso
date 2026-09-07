from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from src.runtime.dataset_catalog import ReplayDatasetCatalog


def _record(idx: int) -> dict:
    return {
        "id": f"event-{idx}",
        "timestamp": datetime(2026, 9, 7, 10, idx, tzinfo=timezone.utc).isoformat(),
        "agent": {"id": "001", "name": "demo-agent"},
        "rule": {"id": "5710", "level": 7, "groups": ["authentication_failed"]},
    }


def _write_jsonl(path: Path, count: int) -> None:
    opener = gzip.open if path.name.endswith(".gz") else path.open
    if path.name.endswith(".gz"):
        handle = opener(path, "wt", encoding="utf-8")
    else:
        handle = opener("w", encoding="utf-8")
    with handle as stream:
        for idx in range(count):
            stream.write(json.dumps(_record(idx)) + "\n")


def test_catalog_inspects_jsonl_and_gzip_with_provenance(tmp_path: Path):
    _write_jsonl(tmp_path / "plain.jsonl", 2)
    _write_jsonl(tmp_path / "compressed.jsonl.gz", 3)
    catalog = ReplayDatasetCatalog(tmp_path)

    items = catalog.list()

    assert [item["name"] for item in items] == ["compressed.jsonl.gz", "plain.jsonl"]
    compressed = items[0]
    assert compressed["total_events"] == 3
    assert compressed["valid_events"] == 3
    assert compressed["invalid_events"] == 0
    assert compressed["compression"] == "gzip"
    assert len(compressed["sha256"]) == 64
    assert compressed["timestamp_start"].startswith("2026-09-07T10:00")
    assert compressed["timestamp_end"].startswith("2026-09-07T10:02")
    assert compressed["classification"] == "unclassified"

    with catalog.open_text("compressed.jsonl.gz") as stream:
        assert len([line for line in stream if line.strip()]) == 3


def test_catalog_cache_reuse_and_invalidation(tmp_path: Path):
    dataset = tmp_path / "demo.jsonl"
    _write_jsonl(dataset, 2)
    catalog = ReplayDatasetCatalog(tmp_path)

    first = catalog.list()[0]
    second = catalog.list()[0]
    assert first["cache_status"] == "refreshed"
    assert second["cache_status"] == "cached"
    assert second["sha256"] == first["sha256"]

    with dataset.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_record(3)) + "\n")

    changed = catalog.list()[0]
    assert changed["cache_status"] == "refreshed"
    assert changed["total_events"] == 3
    assert changed["sha256"] != first["sha256"]


def test_catalog_reads_explicit_classification_without_guessing_filename(tmp_path: Path):
    dataset = tmp_path / "golden-looking-name.jsonl"
    _write_jsonl(dataset, 1)
    catalog = ReplayDatasetCatalog(tmp_path)

    assert catalog.list()[0]["classification"] == "unclassified"

    (tmp_path / "golden-looking-name.jsonl.manifest.json").write_text(
        json.dumps({"classification": "golden", "random_seed": 42}),
        encoding="utf-8",
    )
    classified = catalog.list()[0]
    assert classified["classification"] == "golden"
    assert classified["random_seed"] == 42


def test_catalog_rejects_traversal_and_non_dataset_extension(tmp_path: Path):
    catalog = ReplayDatasetCatalog(tmp_path)
    for invalid in ("../secret.jsonl", "nested/file.jsonl", "notes.txt"):
        try:
            catalog.resolve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected catalog to reject {invalid}")


def test_fast_list_never_inspects_uncached_dataset(tmp_path: Path, monkeypatch):
    _write_jsonl(tmp_path / "large.jsonl", 2)
    catalog = ReplayDatasetCatalog(tmp_path)

    monkeypatch.setattr(catalog, "_inspect", lambda _path: (_ for _ in ()).throw(AssertionError("must not scan")))
    items = catalog.list_fast()

    assert items[0]["name"] == "large.jsonl"
    assert items[0]["inspection_status"] == "pending"
    assert items[0]["total_events"] == 0
    assert items[0]["is_valid"] is False


def test_refresh_all_populates_fast_cache_and_reports_progress(tmp_path: Path):
    _write_jsonl(tmp_path / "a.jsonl", 2)
    _write_jsonl(tmp_path / "b.jsonl", 3)
    catalog = ReplayDatasetCatalog(tmp_path)
    progress = []

    result = catalog.refresh_all(lambda completed, total, name, error: progress.append((completed, total, name, error)))

    assert result["status"] == "COMPLETED"
    assert result["completed_files"] == 2
    assert result["failed_files"] == 0
    assert [item["inspection_status"] for item in catalog.list_fast()] == ["cached", "cached"]
    assert [item["total_events"] for item in catalog.list_fast()] == [2, 3]
    assert progress[-1][:3] == (2, 2, "b.jsonl")
