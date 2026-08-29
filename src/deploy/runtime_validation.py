#!/usr/bin/env python3
"""Container-side deployment runtime environment and artifact validator.

Validates the actual mounted runtime environment as the container user (UID 10001):
1. UID check (assert 10001 when running on POSIX container)
2. Model bundle registry load, metadata truth, and 7-feature schema verification
3. Replay dataset readiness: discovers *.jsonl, validates non-empty, checks first event canonicalization, proves read-only behavior, rejects compressed-only parts
4. State directory behavioral RW proof: write, flush, fsync, atomic rename, read, delete
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.etl.wazuh_canonicalizer import canonicalize_wazuh_alert
from src.model.registry import ModelRegistry, ModelRegistryError


class RuntimeValidationError(Exception):
    """Raised when runtime environment, mounts, permissions, or artifacts fail validation."""
    pass


def validate_user_identity(expected_uid: int = 10001, strict: bool = True) -> bool:
    """Validate runtime effective user ID in container environments."""
    if hasattr(os, "geteuid"):
        current_uid = os.geteuid()
        if current_uid != expected_uid:
            if strict:
                raise RuntimeValidationError(
                    f"Runtime user verification failed: expected UID {expected_uid}, got {current_uid}"
                )
            return False
    return True


def validate_model_artifacts(
    models_dir: Path,
    model_version: str,
) -> Dict[str, str]:
    """Validate model bundle presence, checksums, metadata, and 7-feature schema."""
    if not model_version or not model_version.strip():
        raise RuntimeValidationError("RBTA_MODEL_VERSION environment variable is required and cannot be empty")

    if not models_dir.exists() or not models_dir.is_dir():
        raise RuntimeValidationError(f"Model registry directory does not exist: '{models_dir}'")

    version_dir = models_dir / model_version
    if not version_dir.exists():
        raise RuntimeValidationError(f"Model version directory not found: '{version_dir}'")

    try:
        registry = ModelRegistry(base_dir=models_dir, explicit_version=model_version)
        bundle = registry.load_bundle(model_version)
    except ModelRegistryError as exc:
        raise RuntimeValidationError(f"ModelRegistry bundle load failed for version '{model_version}': {exc}") from exc
    except Exception as exc:
        raise RuntimeValidationError(f"Unexpected error loading model bundle '{model_version}': {exc}") from exc

    # Validate metadata version truth
    loaded_version = bundle.metadata.get("model_version")
    if loaded_version != model_version:
        raise RuntimeValidationError(
            f"Model metadata version mismatch: loaded metadata['model_version']='{loaded_version}' != '{model_version}'"
        )

    # Validate 7-feature schema
    features = bundle.schema.get("features", [])
    if len(features) != 7:
        raise RuntimeValidationError(
            f"Invalid feature schema: expected 7 canonical features, got {len(features)}: {features}"
        )

    return {
        "model_version": model_version,
        "features_count": str(len(features)),
        "tukey_threshold": str(getattr(bundle.threshold, "threshold", "UNKNOWN")),
    }


def validate_replay_datasets(
    replay_dir: Path,
    verify_read_only: bool = True,
) -> Dict[str, str]:
    """Validate replay archive directory, JSONL presence, first-record canonicalization, and read-only mount."""
    if not replay_dir.exists() or not replay_dir.is_dir():
        raise RuntimeValidationError(f"Replay archive directory does not exist: '{replay_dir}'")

    # Discover candidate files
    all_files = [p for p in replay_dir.iterdir() if p.is_file()]
    jsonl_files = [p for p in all_files if p.name.endswith(".jsonl")]
    compressed_files = [
        p for p in all_files
        if p.name.endswith(".gz") or p.name.endswith(".part") or ".jsonl.gz" in p.name
    ]

    if not jsonl_files:
        if compressed_files:
            raise RuntimeValidationError(
                f"Replay directory '{replay_dir}' contains compressed archive parts ({[f.name for f in compressed_files]}), "
                "but no ready *.jsonl dataset. Compressed archives must be derived into replay *.jsonl before deployment."
            )
        raise RuntimeValidationError(
            f"Replay directory '{replay_dir}' contains zero *.jsonl datasets. At least one non-empty *.jsonl dataset is required."
        )

    # Validate non-empty file and first event canonicalization
    valid_file: Optional[Path] = None
    for p in jsonl_files:
        if p.stat().st_size > 0:
            valid_file = p
            break

    if valid_file is None:
        raise RuntimeValidationError(
            f"All discovered *.jsonl files in '{replay_dir}' are empty. At least one non-empty dataset is required."
        )

    # Read and canonicalize first non-empty line
    first_record_ok = False
    with open(valid_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                raw_payload = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise RuntimeValidationError(
                    f"First record in '{valid_file.name}' is not valid JSON: {exc}"
                ) from exc

            try:
                canon_alert = canonicalize_wazuh_alert(raw_payload)
                if not canon_alert.wazuh_alert_id:
                    raise ValueError("Canonicalized alert has empty wazuh_alert_id")
            except Exception as exc:
                raise RuntimeValidationError(
                    f"First record in '{valid_file.name}' failed canonicalization: {exc}"
                ) from exc

            first_record_ok = True
            break

    if not first_record_ok:
        raise RuntimeValidationError(f"File '{valid_file.name}' contained no non-empty JSON lines.")

    # Behavioral test for Read-Only mount if requested
    if verify_read_only and hasattr(os, "geteuid"):
        probe_path = replay_dir / f"__rbta_ro_write_probe_{uuid4()}__"
        try:
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write("illegal write on read-only mount\n")
            # If write succeeded, fail validation
            try:
                probe_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeValidationError(
                f"Replay directory '{replay_dir}' is writable! Replay archive MUST be mounted read-only (:ro)."
            )
        except (PermissionError, OSError):
            # Write rejection is the expected, correct behavior
            pass

    return {
        "dataset_count": str(len(jsonl_files)),
        "first_dataset": valid_file.name,
        "first_dataset_size_bytes": str(valid_file.stat().st_size),
    }


def validate_state_directory_rw(state_dir: Path) -> None:
    """Behaviorally prove that state directory supports write, flush, fsync, atomic rename, read, and delete."""
    if not state_dir.exists():
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeValidationError(f"Cannot create state directory '{state_dir}': {exc}") from exc

    probe_id = str(uuid4())
    temp_file = state_dir / f"__rbta_probe_tmp_{probe_id}.tmp"
    final_file = state_dir / f"__rbta_probe_final_{probe_id}.dat"
    test_bytes = f"rbta_state_rw_proof_{probe_id}\n".encode("utf-8")

    try:
        # 1. Create and write bytes
        with open(temp_file, "wb") as f:
            f.write(test_bytes)
            f.flush()
            os.fsync(f.fileno())

        # 2. Atomic rename
        temp_file.replace(final_file)

        # 3. Read back and verify
        read_back = final_file.read_bytes()
        if read_back != test_bytes:
            raise RuntimeValidationError(
                f"State directory roundtrip data mismatch in '{state_dir}': expected {test_bytes}, got {read_back}"
            )
    except Exception as exc:
        raise RuntimeValidationError(
            f"State directory behavioral RW verification failed in '{state_dir}': {exc}"
        ) from exc
    finally:
        # 4. Cleanup probe files safely
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            final_file.unlink(missing_ok=True)
        except Exception:
            pass


def run_runtime_validation(
    models_dir: Optional[Path] = None,
    model_version: Optional[str] = None,
    replay_dir: Optional[Path] = None,
    state_dir: Optional[Path] = None,
    strict_uid: bool = True,
    verify_ro: bool = True,
) -> Dict[str, Any]:
    """Execute full container runtime readiness validation checklist."""
    models_dir = models_dir or Path(os.environ.get("RBTA_MODEL_REGISTRY_DIR", "/app/artifacts/models")).resolve()
    model_version = model_version or os.environ.get("RBTA_MODEL_VERSION", "")
    replay_dir = replay_dir or Path(os.environ.get("RBTA_REPLAY_DATA_DIR", "/app/data/replay")).resolve()
    state_dir = state_dir or Path(os.environ.get("RBTA_STATE_DIR", os.environ.get("RBTA_STATE_FILE", "/app/data/runtime/state.json"))).resolve()
    if state_dir.is_file() or state_dir.suffix == ".json":
        state_dir = state_dir.parent

    # 1. User UID Check
    validate_user_identity(expected_uid=10001, strict=strict_uid)

    # 2. Model Bundle Validation
    model_info = validate_model_artifacts(models_dir=models_dir, model_version=model_version)

    # 3. Replay Datasets Validation
    replay_info = validate_replay_datasets(replay_dir=replay_dir, verify_read_only=verify_ro)

    # 4. State Directory RW Behavioral Proof
    validate_state_directory_rw(state_dir=state_dir)

    return {
        "status": "PASS",
        "model": model_info,
        "replay": replay_info,
        "state_dir": str(state_dir),
    }


def main() -> int:
    """CLI entrypoint for container execution via `python -m src.deploy.runtime_validation`."""
    try:
        results = run_runtime_validation()
        print("==============================================================================")
        print("✓ CONTAINER RUNTIME VALIDATION PASSED")
        print("==============================================================================")
        print(f"UID Check:              UID 10001 verified")
        print(f"Model Artifact:         version='{results['model']['model_version']}' (7 features, loaded & readable)")
        print(f"Replay Archive:         {results['replay']['dataset_count']} *.jsonl datasets (first='{results['replay']['first_dataset']}', canonicalized & RO)")
        print(f"Runtime State:          '{results['state_dir']}' (RW, fsync, atomic rename proven)")
        print("==============================================================================")
        return 0
    except RuntimeValidationError as exc:
        print(f"FAIL: Runtime validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAIL: Unexpected runtime validation failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
