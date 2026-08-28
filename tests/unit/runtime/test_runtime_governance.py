"""Governance tests for runtime package (Sprint 7)."""
from pathlib import Path

RUNTIME_SRC = Path(__file__).resolve().parent.parent.parent.parent / "src" / "runtime"


def test_shared_core_used_in_runtime():
    """Verify that runtime modules do not duplicate RBTA or feature extraction logic."""
    for py_file in RUNTIME_SRC.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "class RBTAEngine" not in content  # must import, not declare
        assert "class SevenFeatureExtractor" not in content
