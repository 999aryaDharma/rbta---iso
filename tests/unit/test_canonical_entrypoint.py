"""Unit tests for the canonical entrypoint to ensure no legacy imports."""

import importlib
import sys
from pathlib import Path
import pytest

def test_main_imports_no_legacy():
    """Verify that main.py and src.research.orchestrator do not import from src.engine.*."""
    import main
    import src.research.orchestrator as orchestrator

    # Check that orchestrator module has required canonical functions
    assert hasattr(orchestrator, "run_canonical_research_pipeline")
    assert hasattr(orchestrator, "_generate_engineering_smoke_fixture")
    assert hasattr(orchestrator, "main")
    assert hasattr(main, "main")

    # Verify no legacy engine modules were loaded during import
    new_legacy_modules = [m for m in sys.modules if m.startswith("src.engine")]
    invalid_modules = [m for m in new_legacy_modules if not m.startswith("src.rbta.engine") and m != "src.engine"]
    assert len(invalid_modules) == 0, f"Found legacy modules imported: {invalid_modules}"

def test_no_active_code_imports_legacy():
    """Verify that no active python files import from src.engine.*."""
    repo_root = Path(__file__).parent.parent.parent
    src_dir = repo_root / "src"

    invalid_imports = []

    for py_file in src_dir.rglob("*.py"):
        if "archive" in py_file.parts:
            continue

        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("import src.engine") or line.startswith("from src.engine"):
                invalid_imports.append(f"{py_file.relative_to(repo_root)}:{i+1}: {line}")

    assert not invalid_imports, "Found legacy imports in active code:\n" + "\n".join(invalid_imports)


def test_src_never_imports_main():
    """Verify that no file in src/ imports from root main.py."""
    repo_root = Path(__file__).parent.parent.parent
    src_dir = repo_root / "src"

    forbidden_imports = []
    for py_file in src_dir.rglob("*.py"):
        if "archive" in py_file.parts:
            continue
        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("import main") or line.startswith("from main import"):
                forbidden_imports.append(f"{py_file.relative_to(repo_root)}:{i+1}: {line}")

    assert not forbidden_imports, "Found forbidden imports from root main in src/:\n" + "\n".join(forbidden_imports)


def test_main_is_thin_adapter():
    """Verify that root main.py is a thin adapter delegating to src.research.orchestrator."""
    repo_root = Path(__file__).parent.parent.parent
    main_file = repo_root / "main.py"

    content = main_file.read_text(encoding="utf-8")
    assert "src.research.orchestrator" in content
    assert "main()" in content
    # main.py must be a thin adapter (< 30 lines)
    non_empty_lines = [l for l in content.splitlines() if l.strip()]
    assert len(non_empty_lines) <= 20, f"main.py has {len(non_empty_lines)} lines, expected <= 20"

