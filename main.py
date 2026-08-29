"""Canonical Research Orchestrator for RBTA + Isolation Forest.

Executes the end-to-end research methodology pipeline across all authoritative phases:
Phase 1: Ingestion & Canonicalization
Phase 2: RBTA Temporal Aggregation (Agent-Local ETW)
Phase 3: Seven Canonical Feature Extraction
Phase 4: Isolation Forest Reference Training & Model Publication
Phase 5: Stream-Safe Anomaly Scoring & Decision Matrix
Phase 6: Phase A Evaluation (Sensitivity, Baseline, Noise, Complexity)
Phase 7: Phase B Evaluation (Structural Silhouette vs Permutations)
Phase 8: Structured Run Artifact Publication
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence
import uuid

import numpy as np
import pandas as pd

from src.contracts.raw_alert import CanonicalRawAlert
from src.etl.wazuh_canonicalizer import canonicalize_wazuh_alert
from src.evaluation.fixed_window_baseline import run_fixed_window_baseline
from src.evaluation.metrics import compute_arr
from src.evaluation.noise_robustness import run_noise_robustness_evaluation
from src.evaluation.runtime_complexity import run_runtime_complexity_evaluation
from src.evaluation.sensitivity import run_delta_t_sensitivity_analysis
from src.evaluation.structural_silhouette import run_structural_silhouette_evaluation
from src.features.extractor import FEATURE_COLUMNS, SevenFeatureExtractor
from src.model.registry import ModelRegistry
from src.model.scoring_pipeline import (
    ModelArtifactBundle,
    ScoringPipeline,
    train_reference_pipeline,
)
from src.rbta.engine import RBTAEngine
from src.runners.batch_runner import BatchResearchRunner


def _generate_synthetic_research_fixture(n_alerts: int = 250, seed: int = 42) -> List[CanonicalRawAlert]:
    """Generate deterministic synthetic raw alerts for smoke tests and research demonstrations."""
    rng = np.random.default_rng(seed)
    base_t = datetime(2026, 8, 28, 8, 0, 0, tzinfo=timezone.utc)
    agents = [("001", "soc-srv1", 3), ("002", "soc-srv2", 2), ("003", "soc-db", 4), ("004", "soc-gw", 1)]
    rule_groups = ["pam", "sshd", "web", "syslog", "firewall", "ids"]

    alerts: List[CanonicalRawAlert] = []
    current_time = base_t

    # 1. Background standard traffic (mostly benign, low severity)
    for i in range(n_alerts - 30):
        gap_sec = float(rng.exponential(scale=30.0) + 2.0)
        current_time += timedelta(seconds=gap_sec)

        agent_id, agent_name, agent_crit = agents[int(rng.integers(0, len(agents)))]
        group = rule_groups[int(rng.integers(0, len(rule_groups)))]
        sev = int(rng.integers(1, 6))  # Low-medium severity
        rule_id = f"{int(rng.integers(1000, 5000))}"
        mitre = ()

        alerts.append(
            CanonicalRawAlert(
                wazuh_alert_id=f"alert_{i+1:05d}",
                timestamp=current_time,
                agent_id=agent_id,
                agent_name=agent_name,
                rule_group_primary=group,
                rule_level=sev,
                rule_id=rule_id,
                mitre_tactics=mitre,
                srcip=None,
                agent_criticality=agent_crit,
            )
        )

    # 2. Burst of severe anomalous attack events (high severity, multiple critical MITRE tactics)
    attack_time = current_time + timedelta(minutes=10)
    for j in range(30):
        attack_time += timedelta(seconds=float(rng.uniform(1.0, 5.0)))
        alerts.append(
            CanonicalRawAlert(
                wazuh_alert_id=f"alert_{n_alerts - 30 + j + 1:05d}",
                timestamp=attack_time,
                agent_id="003",  # High criticality agent
                agent_name="soc-db",
                rule_group_primary="pam",
                rule_level=14,  # High severity
                rule_id=f"{9000 + (j % 5)}",
                mitre_tactics=("Initial Access", "Execution", "Privilege Escalation"),
                srcip="10.0.0.99",
                agent_criticality=4,
            )
        )

    return alerts


def run_canonical_research_pipeline(
    raw_alerts: Optional[Sequence[CanonicalRawAlert]] = None,
    raw_file_path: Optional[Path] = None,
    output_base_dir: Path = Path("artifacts/research-runs"),
    model_version: str = "rbta-if-canonical-v1",
    base_delta_t_minutes: int = 15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute the canonical research orchestrator pipeline and save structured run artifacts."""
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    run_dir = output_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("=" * 70)
    print(f"RBTA + ISOLATION FOREST CANONICAL RESEARCH PIPELINE")
    print(f"Run ID        : {run_id}")
    print(f"Output Dir    : {run_dir.resolve()}")
    print(f"Model Version : {model_version}")
    print("=" * 70)

    # Phase 1: Ingestion & Canonicalization
    print("\n[Phase 1] Ingestion & Canonicalization...")
    alerts: List[CanonicalRawAlert] = []
    if raw_alerts is not None:
        alerts = list(raw_alerts)
    elif raw_file_path is not None and raw_file_path.exists():
        with raw_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    hit = json.loads(line)
                    alerts.append(canonicalize_wazuh_alert(hit))
    else:
        print("  Using deterministic synthetic test fixture (250 alerts)...")
        alerts = _generate_synthetic_research_fixture(n_alerts=250, seed=random_seed)

    n_raw = len(alerts)
    print(f"  Canonical raw alerts loaded: {n_raw}")

    # Phase 2: RBTA Batch Aggregation
    print("\n[Phase 2] RBTA Temporal Aggregation (Agent-Local ETW)...")
    base_delta_t = timedelta(minutes=base_delta_t_minutes)
    runner = BatchResearchRunner(base_delta_t=base_delta_t, adaptive=True)
    agg_result = runner.run(alerts)
    meta_alerts = agg_result.meta_alerts
    n_meta = len(meta_alerts)
    arr = compute_arr(n_raw, n_meta)
    print(f"  Aggregated MetaAlerts: {n_meta}")
    print(f"  Alert Reduction Rate (ARR): {arr:.2f}%")

    # Phase 3: Feature Extraction
    print("\n[Phase 3] Seven Canonical Feature Extraction...")
    df_features = SevenFeatureExtractor.extract_features_df(meta_alerts)
    print(f"  Extracted feature matrix shape: {df_features.shape}")
    print(f"  Feature columns: {list(FEATURE_COLUMNS)}")

    # Phase 4: Isolation Forest Model Training
    print("\n[Phase 4] Isolation Forest Training & Artifact Publication...")
    bundle = train_reference_pipeline(meta_alerts, random_state=random_seed, model_version=model_version)
    models_dir = run_dir / "models"
    registry = ModelRegistry(base_dir=models_dir)
    published_dir = registry.publish_bundle(bundle, model_version=model_version)
    print(f"  Published model bundle to: {published_dir}")
    print(f"  Tukey Threshold (theta)  : {bundle.threshold.threshold:.4f} (Q3={bundle.threshold.q3:.4f}, IQR={bundle.threshold.iqr:.4f})")

    # Phase 5: Online/Batch Scoring & Decision Matrix
    print("\n[Phase 5] Anomaly Scoring & Decision Matrix Evaluation...")
    pipeline = ScoringPipeline(bundle)
    df_scored, scored_meta_alerts = pipeline.score_meta_alerts(meta_alerts)
    scored_csv_path = run_dir / "meta_alerts_scored.csv"
    df_scored.to_csv(scored_csv_path, index=False)
    print(f"  Scored results exported to: {scored_csv_path}")

    # Phase 6: Phase A Evaluation
    print("\n[Phase 6] Phase A RBTA Evaluations...")
    print("  Running Delta-t Sensitivity Analysis...")
    sens_result = run_delta_t_sensitivity_analysis(alerts)
    print(f"    Recommended Elbow Delta-t: {sens_result.recommended_elbow_delta_t} minutes")

    print("  Running Fixed Tumbling Window Baseline...")
    baseline_result = run_fixed_window_baseline(alerts, window_duration=base_delta_t)
    print(f"    Fixed Window Baseline ARR: {baseline_result.arr:.2f}% (vs RBTA: {arr:.2f}%)")

    print("  Running Noise Robustness Evaluation...")
    noise_result = run_noise_robustness_evaluation(alerts, delta_t=base_delta_t, random_seed=random_seed)

    print("  Running Runtime Complexity Evaluation...")
    complexity_result = run_runtime_complexity_evaluation(alerts, n_subsets=6, delta_t=base_delta_t)
    print(f"    Runtime O(n log k) Linear Fit R^2: {complexity_result.r_squared:.4f} (Slope: {complexity_result.slope:.6f} ms/alert)")

    phase_a_summary = {
        "rbta_arr": arr,
        "fixed_baseline_arr": baseline_result.arr,
        "arr_advantage_percent_points": round(arr - baseline_result.arr, 2),
        "recommended_elbow_delta_t_minutes": sens_result.recommended_elbow_delta_t,
        "runtime_r_squared": complexity_result.r_squared,
        "mean_throughput_alerts_per_ms": complexity_result.mean_throughput,
        "sensitivity_curve": sens_result.summary_df.to_dict(orient="records"),
        "noise_robustness": noise_result.summary_df.to_dict(orient="records"),
        "complexity_subsets": complexity_result.subset_df.to_dict(orient="records"),
    }
    with (run_dir / "phase_a_results.json").open("w", encoding="utf-8") as f:
        json.dump(phase_a_summary, f, indent=2)

    # Phase 7: Phase B Evaluation
    print("\n[Phase 7] Phase B Structural Silhouette Evaluation...")
    silhouette_result = run_structural_silhouette_evaluation(
        scored_meta_alerts, bundle, n_permutations=100, random_seed=random_seed
    )
    phase_b_summary = {
        "is_calculable": silhouette_result.is_calculable,
        "uncalculable_reason": silhouette_result.uncalculable_reason,
        "observed_silhouette": silhouette_result.observed_silhouette,
        "null_distribution_mean": silhouette_result.random_mean,
        "null_distribution_std": silhouette_result.random_std,
        "null_distribution_min": silhouette_result.random_min,
        "null_distribution_max": silhouette_result.random_max,
        "observed_percentile": silhouette_result.observed_percentile,
        "z_score": silhouette_result.z_score,
        "empirical_p_value": silhouette_result.empirical_p_value,
        "n_permutations": silhouette_result.n_valid_permutations,
    }
    with (run_dir / "phase_b_results.json").open("w", encoding="utf-8") as f:
        json.dump(phase_b_summary, f, indent=2)

    if silhouette_result.is_calculable:
        print(f"  Observed Silhouette Score : {silhouette_result.observed_silhouette:.4f}")
        print(f"  Null Distribution Mean    : {silhouette_result.random_mean:.4f} +/- {silhouette_result.random_std:.4f}")
        print(f"  Standardized Z-Score      : {silhouette_result.z_score:.2f}")
        print(f"  Empirical p-value         : {silhouette_result.empirical_p_value:.4f}")
    else:
        print(f"  Silhouette evaluation not calculable: {silhouette_result.uncalculable_reason}")

    # Phase 8: Structured Run Artifact Publication
    elapsed_total = time.perf_counter() - t_start
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_total, 3),
        "n_raw_alerts": n_raw,
        "n_meta_alerts": n_meta,
        "arr": arr,
        "model_version": model_version,
        "random_seed": random_seed,
        "base_delta_t_minutes": base_delta_t_minutes,
        "published_artifacts": {
            "phase_a_results": "phase_a_results.json",
            "phase_b_results": "phase_b_results.json",
            "meta_alerts_scored": "meta_alerts_scored.csv",
            "research_summary": "research_summary.json",
        },
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    research_summary = {
        "manifest": manifest,
        "phase_a": phase_a_summary,
        "phase_b": phase_b_summary,
    }
    with (run_dir / "research_summary.json").open("w", encoding="utf-8") as f:
        json.dump(research_summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"CANONICAL RESEARCH PIPELINE COMPLETED IN {elapsed_total:.2f}s")
    print(f"All artifacts published to: {run_dir.resolve()}")
    print("=" * 70)

    return research_summary


def main() -> None:
    """CLI entrypoint for running the research orchestrator."""
    parser = argparse.ArgumentParser(
        description="RBTA + Isolation Forest Canonical Research Orchestrator"
    )
    parser.add_argument("--input", type=Path, default=None, help="Path to raw JSONL Wazuh alerts file")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research-runs"), help="Output directory")
    parser.add_argument("--model-version", type=str, default="rbta-if-canonical-v1", help="Model version identifier")
    parser.add_argument("--delta-t", type=int, default=15, help="Base Delta-t window in minutes (default 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")

    args = parser.parse_args()
    run_canonical_research_pipeline(
        raw_file_path=args.input,
        output_base_dir=args.output_dir,
        model_version=args.model_version,
        base_delta_t_minutes=args.delta_t,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
