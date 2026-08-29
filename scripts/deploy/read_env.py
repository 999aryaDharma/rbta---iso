#!/usr/bin/env python3
"""Deterministic, standard-library-only environment file parser for RBTA deployment harness."""

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, Optional


def parse_env_file(filepath: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE .env file deterministically without shell evaluation."""
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    env_map: Dict[str, str] = {}
    lines = filepath.read_text(encoding="utf-8").splitlines()

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip blank lines and full-line comments
        if not stripped or stripped.startswith("#"):
            continue

        # Reject lines that don't match KEY=VALUE
        if "=" not in line:
            raise ValueError(f"Malformed line {idx} in {filepath}: missing '=' delimiter: '{line}'")

        key_part, val_part = line.split("=", 1)
        key = key_part.strip()

        # Validate key identifier
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"Invalid key identifier '{key}' on line {idx} in {filepath}")

        # Clean value: strip leading/trailing whitespace
        val = val_part.strip()

        # Remove surrounding matching quotes if present
        if len(val) >= 2 and ((val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'"))):
            val = val[1:-1]

        env_map[key] = val

    return env_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic .env parser for RBTA deployment scripts.")
    parser.add_argument("env_file", type=Path, help="Path to .env file")
    parser.add_argument("key", nargs="?", default=None, help="Specific key to retrieve")
    parser.add_argument("--default", default=None, help="Default value if key is not found")
    parser.add_argument("--require", action="store_true", help="Fail if key is missing or empty")
    parser.add_argument("--json", action="store_true", help="Output parsed environment as JSON")
    args = parser.parse_args()

    try:
        env_map = parse_env_file(args.env_file)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps(env_map, indent=2))
        return 0

    if args.key:
        val = env_map.get(args.key, args.default)
        if args.require and (val is None or val == ""):
            print(f"ERROR: Required key '{args.key}' is missing or empty in {args.env_file}", file=sys.stderr)
            return 1
        if val is not None:
            print(val)
        return 0

    # If no key specified, output as KEY=VALUE lines
    for k, v in env_map.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
