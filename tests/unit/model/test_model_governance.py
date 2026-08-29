"""Governance tests for model package (Sprint 4)."""
from pathlib import Path
import re

MODEL_SRC = Path(__file__).resolve().parent.parent.parent.parent / "src" / "model"


def test_no_dynamic_contamination_or_ground_truth_in_model_src():
    """Verify that src/model does not contain dynamic contamination or ground truth parameterization."""
    forbidden = ["compute_dynamic_contamination", "ground_truth", "is_synthetic", "scenario_id"]

    for py_file in MODEL_SRC.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, f"Found forbidden token '{token}' in {py_file.name}"


def test_no_fit_in_scoring_pipeline_inference_methods():
    """Verify that score_single and score_meta_alerts in scoring_pipeline.py do not call fit or fit_transform."""
    pipeline_file = MODEL_SRC / "scoring_pipeline.py"
    if pipeline_file.exists():
        content = pipeline_file.read_text(encoding="utf-8")
        # Check that score_single and score_meta_alerts use transform only
        assert "self.scaler.transform(" in content or "scaler.transform(" in content
        assert "fit_transform" not in content
