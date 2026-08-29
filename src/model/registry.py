"""Model artifact registry with atomic staging publication."""

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Union
import uuid
import joblib

from src.features.extractor import FEATURE_COLUMNS
from src.model.calibration import ScoreCalibration
from src.model.scoring_pipeline import ModelArtifactBundle
from src.model.threshold import TukeyThreshold


class ModelRegistryError(RuntimeError):
    """Raised when artifact publication, validation, or loading fails."""
    pass


REQUIRED_ARTIFACT_FILES = (
    "isolation_forest.joblib",
    "robust_scaler.joblib",
    "score_calibration.json",
    "threshold.json",
    "feature_schema.json",
    "metadata.json",
)


class ModelRegistry:
    """Manages versioned model artifacts with atomic staging publication."""

    def __init__(self, base_dir: Union[str, Path] = "artifacts/models", explicit_version: Optional[str] = None) -> None:
        self.base_dir: Path = Path(base_dir).resolve()
        self.staging_dir: Path = self.base_dir / ".staging"
        self._explicit_version = explicit_version or os.getenv("RBTA_MODEL_VERSION")

    def get_active_version(self) -> Optional[str]:
        """Discover configured active model version (explicit constructor or RBTA_MODEL_VERSION env var)."""
        candidate_version = self._explicit_version or os.getenv("RBTA_MODEL_VERSION")
        if candidate_version:
            version_dir = self.base_dir / candidate_version
            if version_dir.exists() and version_dir.is_dir():
                return candidate_version
        return None

    def publish_bundle(self, bundle: ModelArtifactBundle, model_version: str) -> Path:
        """Publish a model bundle atomically via staging.

        Parameters
        ----------
        bundle : ModelArtifactBundle
            Bundle to serialize and publish.
        model_version : str
            Target model version directory name.

        Returns
        -------
        Path
            Path to the published model directory.

        Raises
        ------
        ModelRegistryError
            If validation or publication fails.
        """
        target_dir = self.base_dir / model_version
        if target_dir.exists():
            raise ModelRegistryError(f"Model version directory '{target_dir}' already exists")

        stage_id = str(uuid.uuid4())
        stage_dir = self.staging_dir / stage_id
        stage_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Write all 6 files
            joblib.dump(bundle.model, stage_dir / "isolation_forest.joblib")
            joblib.dump(bundle.scaler, stage_dir / "robust_scaler.joblib")

            with (stage_dir / "score_calibration.json").open("w", encoding="utf-8") as f:
                json.dump(bundle.calibration.to_dict(), f, indent=2)

            with (stage_dir / "threshold.json").open("w", encoding="utf-8") as f:
                json.dump(bundle.threshold.to_dict(), f, indent=2)

            with (stage_dir / "feature_schema.json").open("w", encoding="utf-8") as f:
                json.dump(bundle.schema, f, indent=2)

            with (stage_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(bundle.metadata, f, indent=2)

            manifest = {}
            for fname in REQUIRED_ARTIFACT_FILES:
                fpath = stage_dir / fname
                sha256 = hashlib.sha256(fpath.read_bytes()).hexdigest()
                manifest[fname] = sha256
            with (stage_dir / "manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # 2. Validate roundtrip in staging
            self._validate_directory(stage_dir)

            # 3. Publish atomically
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage_dir), str(target_dir))

            return target_dir

        except Exception as exc:
            if stage_dir.exists():
                shutil.rmtree(str(stage_dir), ignore_errors=True)
            raise ModelRegistryError(f"Failed to publish model bundle: {exc}") from exc

    def load_bundle(self, model_version_or_path: Union[str, Path]) -> ModelArtifactBundle:
        """Load and validate a published model artifact bundle.

        Parameters
        ----------
        model_version_or_path : str | Path
            Version name or direct path to model directory.

        Returns
        -------
        ModelArtifactBundle
            Loaded artifact bundle ready for inference.

        Raises
        ------
        ModelRegistryError
            If directory is missing, incomplete, or schema is invalid.
        """
        model_path = Path(model_version_or_path)
        if not model_path.is_absolute():
            model_path = self.base_dir / model_path

        if not model_path.exists() or not model_path.is_dir():
            raise ModelRegistryError(f"Model directory does not exist: {model_path}")

        self._validate_directory(model_path)

        try:
            model = joblib.load(model_path / "isolation_forest.joblib")
            scaler = joblib.load(model_path / "robust_scaler.joblib")

            with (model_path / "score_calibration.json").open("r", encoding="utf-8") as f:
                cal_data = json.load(f)
            calibration = ScoreCalibration.from_dict(cal_data)

            with (model_path / "threshold.json").open("r", encoding="utf-8") as f:
                thresh_data = json.load(f)
            threshold = TukeyThreshold.from_dict(thresh_data)

            with (model_path / "feature_schema.json").open("r", encoding="utf-8") as f:
                schema_data = json.load(f)

            with (model_path / "metadata.json").open("r", encoding="utf-8") as f:
                metadata_data = json.load(f)

            return ModelArtifactBundle(
                scaler=scaler,
                model=model,
                calibration=calibration,
                threshold=threshold,
                metadata=metadata_data,
                schema=schema_data,
            )

        except Exception as exc:
            raise ModelRegistryError(f"Error reading model artifacts from '{model_path}': {exc}") from exc

    def _validate_directory(self, dir_path: Path) -> None:
        """Ensure all required artifact files exist and feature schema is valid."""
        for filename in REQUIRED_ARTIFACT_FILES:
            file_path = dir_path / filename
            if not file_path.exists():
                raise ModelRegistryError(f"Missing required artifact file '{filename}' in '{dir_path}'")

        # Validate feature schema matches canonical FEATURE_COLUMNS exactly
        schema_file = dir_path / "feature_schema.json"
        with schema_file.open("r", encoding="utf-8") as f:
            schema_data = json.load(f)

        features = tuple(schema_data.get("features", []))
        if features != FEATURE_COLUMNS:
            raise ModelRegistryError(
                f"Feature schema mismatch in '{dir_path}': expected {FEATURE_COLUMNS}, got {features}"
            )

        manifest_file = dir_path / "manifest.json"
        if not manifest_file.exists():
            raise ModelRegistryError(f"Missing mandatory manifest.json in '{dir_path}'")

        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            raise ModelRegistryError(f"Malformed manifest.json in '{dir_path}': {exc}") from exc

        if not isinstance(manifest, dict):
            raise ModelRegistryError(f"Manifest is not a JSON object in '{dir_path}'")

        if set(manifest.keys()) != set(REQUIRED_ARTIFACT_FILES):
            raise ModelRegistryError(
                f"Manifest keys {set(manifest.keys())} != expected {set(REQUIRED_ARTIFACT_FILES)}"
            )

        for fname, expected_hash in manifest.items():
            fpath = dir_path / fname
            if not fpath.exists():
                raise ModelRegistryError(f"Manifest references missing file '{fname}'")
            actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ModelRegistryError(f"Checksum mismatch for '{fname}': expected {expected_hash}, got {actual_hash}")
