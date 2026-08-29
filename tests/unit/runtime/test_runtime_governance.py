"""Governance tests for runtime package (Sprint 7 and S7 No-Drop Remediation)."""

from pathlib import Path

RUNTIME_SRC = Path(__file__).resolve().parent.parent.parent.parent / "src" / "runtime"


def test_shared_core_used_in_runtime():
    """Verify that runtime modules do not duplicate RBTA or feature extraction logic."""
    for py_file in RUNTIME_SRC.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "class RBTAEngine" not in content  # must import, not declare
        assert "class SevenFeatureExtractor" not in content


def test_no_timestamp_drop_logic_in_runtime_src():
    """Verify that runtime production code does not implement timestamp-based event drop logic."""
    forbidden_tokens = [
        "late_drop",
        "max_lateness",
        "drop_late",
        "too_old",
        "expired_alert",
    ]
    for py_file in RUNTIME_SRC.glob("*.py"):
        content = py_file.read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            assert token not in content, f"Forbidden token '{token}' found in {py_file.name}"
