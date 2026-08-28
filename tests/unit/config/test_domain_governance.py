"""Governance tests ensuring no duplicate authoritative domain constants exist."""
import re
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


def test_no_duplicate_authoritative_domain_declarations():
    """Verify that domain constants are defined only in src/config/domain.py."""
    forbidden_declarations = [
        re.compile(r"^AGENT_CRITICALITY\s*(:\s*[^=]+)?\s*=", re.MULTILINE),
        re.compile(r"^GROUP_SEVERITY_WEIGHT\s*(:\s*[^=]+)?\s*=", re.MULTILINE),
        re.compile(r"^CRITICAL_MITRE_TACTICS\s*(:\s*[^=]+)?\s*=", re.MULTILINE),
    ]

    duplicates = []
    for py_file in SRC_DIR.rglob("*.py"):
        rel_path = py_file.relative_to(SRC_DIR).as_posix()
        # Allowed only in config/domain.py
        if rel_path == "config/domain.py":
            continue

        content = py_file.read_text(encoding="utf-8")
        for pattern in forbidden_declarations:
            matches = pattern.findall(content)
            if matches:
                duplicates.append(f"{rel_path}: matches {pattern.pattern}")

    assert not duplicates, (
        f"Found duplicate authoritative domain constant declarations in:\n"
        + "\n".join(duplicates)
        + "\nAll domain constants MUST be imported from src.config.domain."
    )
