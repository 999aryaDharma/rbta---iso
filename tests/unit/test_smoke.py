"""Smoke test to verify project and test harness foundation."""
import importlib
import pytest


def test_python_version_and_imports():
    """Verify that essential standard and third-party libraries import correctly."""
    core_modules = [
        "numpy",
        "pandas",
        "sklearn",
        "joblib",
        "pytest",
    ]
    for mod_name in core_modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Module {mod_name} could not be loaded."


def test_fixtures_available():
    """Verify that fixture files exist and are valid JSON."""
    import json
    from pathlib import Path

    fixture_dir = Path(__file__).resolve().parent.parent / "fixtures" / "wazuh"
    assert fixture_dir.exists(), f"Fixture directory {fixture_dir} does not exist."

    fixture_files = [
        "raw_alert_standard.json",
        "opensearch_hit.json",
        "raw_alert_no_mitre.json",
        "raw_alert_flattened_mitre.json",
    ]
    for fname in fixture_files:
        fpath = fixture_dir / fname
        assert fpath.exists(), f"Fixture file {fname} not found."
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Fixture {fname} did not contain a valid JSON object."
