"""Governance tests ensuring unique authoritative feature schema (Sprint 3)."""
from pathlib import Path
import re

SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


def test_single_authoritative_feature_columns_declaration():
    """Verify that FEATURE_COLUMNS is declared only in src/features/extractor.py."""
    pattern = re.compile(r"^FEATURE_COLUMNS\s*(:\s*[^=]+)?\s*=", re.MULTILINE)
    declarations = []

    for py_file in SRC_DIR.rglob("*.py"):
        rel_path = py_file.relative_to(SRC_DIR).as_posix()
        # Allowed in features/extractor.py
        if rel_path in ("features/extractor.py", "features/__init__.py"):
            continue
        # Skip legacy engine modules until deprecated in later sprints
        if rel_path in ("engine/feature_engineering.py", "engine/isolation_forest.py", "engine/rbta_core.py"):
            continue

        content = py_file.read_text(encoding="utf-8")
        if pattern.search(content):
            declarations.append(rel_path)

    assert not declarations, (
        f"Found duplicate FEATURE_COLUMNS declarations in:\n" + "\n".join(declarations)
    )
