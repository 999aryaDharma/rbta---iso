import gzip
import json
from pathlib import Path

import pytest

from src.research.corpus_input import (
    ResearchCorpusError,
    discover_research_inputs,
    load_research_corpus,
)


def _event(event_id: str, timestamp: str) -> dict:
    return {
        "id": event_id,
        "timestamp": timestamp,
        "agent": {"id": "001", "name": "lenovo-demo"},
        "rule": {"id": "5710", "level": 7, "groups": ["authentication_failed"]},
    }


def _write(path: Path, rows: list[dict]) -> None:
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    else:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")


def test_discovers_daily_jsonl_and_gzip_in_deterministic_order_and_ignores_meta(tmp_path: Path):
    _write(tmp_path / "wazuh-alerts-4.x-2026.04.03.jsonl", [_event("3", "2026-04-03T00:00:00Z")])
    _write(tmp_path / "wazuh-alerts-4.x-2026.04.02.jsonl.gz", [_event("2", "2026-04-02T00:00:00Z")])
    (tmp_path / "wazuh-alerts-4.x-2026.04.02.meta").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    _write(tmp_path / "nested" / "hidden.jsonl", [_event("nested", "2026-04-01T00:00:00Z")])

    files = discover_research_inputs(tmp_path)

    assert [path.name for path in files] == [
        "wazuh-alerts-4.x-2026.04.02.jsonl.gz",
        "wazuh-alerts-4.x-2026.04.03.jsonl",
    ]


def test_loads_directory_as_one_globally_chronological_corpus_with_provenance(tmp_path: Path):
    _write(tmp_path / "b.jsonl", [_event("late", "2026-04-03T00:00:00Z")])
    _write(tmp_path / "a.jsonl", [_event("early", "2026-04-02T00:00:00Z")])

    result = load_research_corpus(tmp_path)

    assert [alert.wazuh_alert_id for alert in result.alerts] == ["early", "late"]
    assert result.provenance["source_kind"] == "directory"
    assert [item["name"] for item in result.provenance["files"]] == ["a.jsonl", "b.jsonl"]
    assert all(len(item["sha256"]) == 64 for item in result.provenance["files"])
    assert len(result.provenance["corpus_sha256"]) == 64


def test_duplicate_alert_id_across_daily_files_fails_to_prevent_temporal_leakage(tmp_path: Path):
    _write(tmp_path / "a.jsonl", [_event("duplicate", "2026-04-02T00:00:00Z")])
    _write(tmp_path / "b.jsonl", [_event("duplicate", "2026-04-03T00:00:00Z")])

    with pytest.raises(ResearchCorpusError, match="Duplicate Wazuh alert id.*a.jsonl:1.*b.jsonl:1"):
        load_research_corpus(tmp_path)


def test_malformed_record_reports_file_and_line(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(_event("ok", "2026-04-02T00:00:00Z")) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ResearchCorpusError, match=r"broken\.jsonl:2"):
        load_research_corpus(path)


def test_empty_or_unsupported_source_fails_closed(tmp_path: Path):
    (tmp_path / "only.meta").write_text("{}", encoding="utf-8")
    with pytest.raises(ResearchCorpusError, match="zero supported"):
        discover_research_inputs(tmp_path)
    with pytest.raises(ResearchCorpusError, match="\.jsonl"):
        discover_research_inputs(tmp_path / "only.meta")
