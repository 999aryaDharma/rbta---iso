"""Unit tests for the canonical entrypoint to ensure no legacy imports."""

import importlib
import sys
from pathlib import Path
import pytest

def test_main_imports_no_legacy():
    """Verify that main.py does not import from src.engine.*."""
    legacy_modules = [m for m in sys.modules if m.startswith("src.engine")]
    
    import main
    
    # Check that main module has required canonical attributes
    assert hasattr(main, "RBTAEngine")
    assert hasattr(main, "SevenFeatureExtractor")
    
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
