"""Unit tests for HistoricalCheckpoint persistence (Sprint 5)."""
from pathlib import Path
import pytest
from src.ingestion.checkpoint import HistoricalCheckpoint, CheckpointManager


def test_checkpoint_defaults_and_updates():
    """Checkpoint initializes with defaults and correctly updates position."""
    cp = HistoricalCheckpoint()
    assert cp.mode == "historical"
    assert cp.current_index is None
    assert cp.last_sort is None
    assert cp.processed_count == 0
    assert cp.completed_indices == []

    cp.update(
        index_name="wazuh-alerts-4.x-2026.04.02",
        last_sort=[1775118766650, "1775118766.393"],
        wazuh_alert_id="1775118766.393",
    )
    assert cp.current_index == "wazuh-alerts-4.x-2026.04.02"
    assert cp.last_sort == [1775118766650, "1775118766.393"]
    assert cp.last_wazuh_alert_id == "1775118766.393"
    assert cp.processed_count == 1

    cp.mark_index_completed("wazuh-alerts-4.x-2026.04.02")
    assert "wazuh-alerts-4.x-2026.04.02" in cp.completed_indices
    assert cp.current_index is None
    assert cp.last_sort is None


def test_checkpoint_file_persistence_roundtrip(tmp_path: Path):
    """CheckpointManager saves to JSON file and restores identically."""
    cp_file = tmp_path / "checkpoint.json"
    manager = CheckpointManager(cp_file)

    # Initial load on missing file returns fresh checkpoint
    cp = manager.load()
    assert cp.processed_count == 0

    cp.update("wazuh-alerts-4.x-2026.04.02", [100, "alert-1"], "alert-1")
    manager.save(cp)

    assert cp_file.exists()

    # Restore from disk
    restored = manager.load()
    assert restored.current_index == "wazuh-alerts-4.x-2026.04.02"
    assert restored.last_sort == [100, "alert-1"]
    assert restored.last_wazuh_alert_id == "alert-1"
    assert restored.processed_count == 1
