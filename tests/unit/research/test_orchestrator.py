"""Tests for canonical research orchestrator (phase order, delta-t propagation, subsets, input modes)."""

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.research.orchestrator import (
    run_canonical_research_pipeline,
    ResearchInputError,
    _generate_engineering_smoke_fixture,
    main,
)


def test_missing_input_never_falls_back_to_fixture():
    """Verify that a missing input file raises ResearchInputError / FileNotFoundError and fails fast."""
    with pytest.raises((ResearchInputError, FileNotFoundError)):
        run_canonical_research_pipeline(raw_file_path=Path("non_existent_file_12345.jsonl"))


def test_fixture_mode_explicitly_marked_non_research(tmp_path):
    """Verify that fixture mode sets research_results_valid_for_seminar=False in metadata."""
    fixture_alerts = _generate_engineering_smoke_fixture(n_alerts=50, seed=42)
    summary = run_canonical_research_pipeline(
        raw_alerts=fixture_alerts,
        output_base_dir=tmp_path,
        is_fixture_mode=True,
        random_seed=42,
    )
    manifest = summary["manifest"]
    assert manifest["input_mode"] == "engineering_fixture"
    assert manifest["research_results_valid_for_seminar"] is False


def test_auto_delta_selection_drives_final_rbta(tmp_path):
    """Verify that in auto mode, recommended elbow delta-t drives final RBTA."""
    fixture_alerts = _generate_engineering_smoke_fixture(n_alerts=60, seed=42)
    summary = run_canonical_research_pipeline(
        raw_alerts=fixture_alerts,
        output_base_dir=tmp_path,
        is_fixture_mode=True,
        delta_t_mode="auto",
        random_seed=42,
    )
    manifest = summary["manifest"]
    assert manifest["delta_t_selection_source"] == "sensitivity_elbow"
    assert manifest["selected_delta_t_minutes"] == manifest["recommended_delta_t_minutes"]
    assert manifest["selected_delta_t_minutes"] > 0


def test_manual_delta_override_is_explicit(tmp_path):
    """Verify that manual delta override is explicitly recorded and drives final RBTA."""
    fixture_alerts = _generate_engineering_smoke_fixture(n_alerts=60, seed=42)
    summary = run_canonical_research_pipeline(
        raw_alerts=fixture_alerts,
        output_base_dir=tmp_path,
        is_fixture_mode=True,
        delta_t_mode=30,
        random_seed=42,
    )
    manifest = summary["manifest"]
    assert manifest["delta_t_selection_source"] == "manual_override"
    assert manifest["selected_delta_t_minutes"] == 30
    assert "recommended_delta_t_minutes" in manifest


def test_runtime_uses_exactly_eight_subsets(tmp_path):
    """Verify that the orchestrator executes runtime complexity proof with exactly 8 subsets."""
    fixture_alerts = _generate_engineering_smoke_fixture(n_alerts=60, seed=42)
    summary = run_canonical_research_pipeline(
        raw_alerts=fixture_alerts,
        output_base_dir=tmp_path,
        is_fixture_mode=True,
        random_seed=42,
    )
    phase_a = summary["phase_a"]
    assert len(phase_a["complexity_subsets"]) == 8


def test_canonical_research_phase_order(tmp_path):
    """Verify strictly ordered execution of research phases."""
    call_order = []

    fixture_alerts = _generate_engineering_smoke_fixture(n_alerts=60, seed=42)

    import src.research.orchestrator as orch

    orig_sens = orch.run_delta_t_sensitivity_analysis
    def spy_sens(*args, **kwargs):
        call_order.append("sensitivity")
        return orig_sens(*args, **kwargs)

    orig_batch_runner = orch.BatchResearchRunner
    def spy_batch_runner(*args, **kwargs):
        call_order.append(f"batch_runner_init_delta_{kwargs.get('base_delta_t')}_adaptive_{kwargs.get('adaptive')}")
        instance = orig_batch_runner(*args, **kwargs)
        orig_run = instance.run
        def spy_run(*r_args, **r_kwargs):
            call_order.append("final_rbta_run")
            return orig_run(*r_args, **r_kwargs)
        instance.run = spy_run
        return instance

    orig_fixed = orch.run_fixed_window_baseline
    def spy_fixed(*args, **kwargs):
        call_order.append("fixed_window")
        return orig_fixed(*args, **kwargs)

    orig_noise = orch.run_noise_robustness_evaluation
    def spy_noise(*args, **kwargs):
        call_order.append("noise")
        return orig_noise(*args, **kwargs)

    orig_runtime = orch.run_runtime_complexity_evaluation
    def spy_runtime(*args, **kwargs):
        call_order.append("runtime")
        return orig_runtime(*args, **kwargs)

    orig_extractor = orch.SevenFeatureExtractor.extract_features_df
    def spy_extractor(*args, **kwargs):
        call_order.append("seven_features")
        return orig_extractor(*args, **kwargs)

    orig_train = orch.train_reference_pipeline
    def spy_train(*args, **kwargs):
        call_order.append("train_model")
        return orig_train(*args, **kwargs)

    orig_scoring_cls = orch.ScoringPipeline
    def spy_scoring_pipeline(*args, **kwargs):
        instance = orig_scoring_cls(*args, **kwargs)
        orig_score_m = instance.score_meta_alerts
        def spy_score_m(metas, *m_args, **m_kwargs):
            call_order.append("scoring")
            return orig_score_m(metas, *m_args, **m_kwargs)
        instance.score_meta_alerts = spy_score_m
        return instance

    orig_silhouette = orch.run_structural_silhouette_evaluation
    def spy_silhouette(*args, **kwargs):
        call_order.append("structural_silhouette")
        return orig_silhouette(*args, **kwargs)

    with patch.object(orch, "run_delta_t_sensitivity_analysis", side_effect=spy_sens), \
         patch.object(orch, "BatchResearchRunner", side_effect=spy_batch_runner), \
         patch.object(orch, "run_fixed_window_baseline", side_effect=spy_fixed), \
         patch.object(orch, "run_noise_robustness_evaluation", side_effect=spy_noise), \
         patch.object(orch, "run_runtime_complexity_evaluation", side_effect=spy_runtime), \
         patch.object(orch.SevenFeatureExtractor, "extract_features_df", side_effect=spy_extractor), \
         patch.object(orch, "train_reference_pipeline", side_effect=spy_train), \
         patch.object(orch, "ScoringPipeline", side_effect=spy_scoring_pipeline), \
         patch.object(orch, "run_structural_silhouette_evaluation", side_effect=spy_silhouette):

        summary = orch.run_canonical_research_pipeline(
            raw_alerts=fixture_alerts,
            output_base_dir=tmp_path,
            is_fixture_mode=True,
            delta_t_mode="auto",
            random_seed=42,
        )

    expected_phases = [
        "sensitivity",
        "final_rbta_run",
        "fixed_window",
        "noise",
        "runtime",
        "train_model",
        "scoring",
        "structural_silhouette",
    ]

    filtered_calls = [c for c in call_order if c in expected_phases]
    assert filtered_calls == expected_phases

    # Verify final RBTA had adaptive == True
    init_call = [c for c in call_order if c.startswith("batch_runner_init")][0]
    assert "adaptive_True" in init_call


