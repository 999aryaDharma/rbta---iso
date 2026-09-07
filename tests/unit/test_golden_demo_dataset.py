import gzip
import json
from pathlib import Path

from scripts.build_golden_demo_dataset import build_golden


def test_build_golden_selects_valid_real_events_deterministically_and_sorts_them(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    events = [
        {
            "id": f"event-{idx}",
            "timestamp": f"2026-08-28T10:{5 - idx:02d}:00+00:00",
            "agent": {"id": "001", "name": "soc"},
            "rule": {"id": "5501", "level": idx + 1, "groups": ["pam"]},
        }
        for idx in range(6)
    ]
    source.write_text("\n".join(json.dumps(item) for item in events) + "\n{broken\n", encoding="utf-8")
    output = tmp_path / "golden.jsonl.gz"

    manifest = build_golden(source, output, sample_size=3, seed=42)

    assert manifest["classification"] == "golden"
    assert manifest["selected_events"] == 3
    assert manifest["source_valid_events"] == 6
    assert manifest["source_invalid_events"] == 1
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        selected = [json.loads(line) for line in stream]
    assert [item["timestamp"] for item in selected] == sorted(item["timestamp"] for item in selected)
    assert json.loads((tmp_path / "golden.jsonl.gz.manifest.json").read_text())["random_seed"] == 42
