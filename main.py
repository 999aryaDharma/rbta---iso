"""
main.py  —  Orchestrator RBTA v5 + IF 13-fitur + Evaluasi Lengkap
==================================================================
Urutan eksekusi:

  STEP 1  preprocessing          → load_and_prepare()
  STEP 2  [opsional] injeksi     → attack_injector.run_injection()
            Diaktifkan via: USE_INJECTED_DATA = True
            Jika True → pakai data/injected/rbta_injected.csv
            Jika False → pakai data/raw/rbta_ready_ALL.csv
  STEP 3  sensitivity analysis   → metrics.sensitivity_analysis()
  STEP 4  RBTA v5 optimal Δt     → rbta_algorithm_02.run_rbta()
  STEP 5  feature engineering    → add_if_features() + enrich_features()
  STEP 6  [opsional] propagasi   → attack_injector.propagate_labels()
            Hanya berjalan jika USE_INJECTED_DATA = True
  STEP 7  fixed window baseline  → fixed_window_baseline.run_fixed_window()
  STEP 8  loss analysis          → fixed_window_baseline.loss_analysis()
  STEP 9  isolation forest       → isolation_forest.run_pipeline()
  STEP 10 evaluasi komprehensif  → metrics.comprehensive_report()
            + plot_pr_curve, plot_fnr_vs_arr (jika ground_truth tersedia)
  STEP 11 plot detail            → reuse cache sensitivity
"""

import gc
import logging
import os
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.etl.preprocessing_01          import load_and_prepare
from src.engine.rbta_algorithm_02       import run_rbta, validate_mapping, add_if_features
from src.engine.fixed_window_baseline   import run_fixed_window, loss_analysis
from src.engine.feature_engineering     import enrich_features
from src.engine.isolation_forest        import run_pipeline as run_isolation_forest
from src.evaluation.attack_injector     import run_injection, propagate_labels
from src.evaluation.metrics             import (
    sensitivity_analysis,
    find_elbow,
    print_sensitivity_table,
    print_enhancement_report,
    plot_sensitivity,
    plot_severity_dist,
    plot_alert_count_dist,
    plot_watermark_stats,
    plot_elastic_dt_history,
    comprehensive_report,
    compute_fnr_arr_tradeoff,
    compute_pr_auc,
    plot_pr_curve,
    plot_fnr_vs_arr,
    DELTA_T_VALUES,
    BUFFER_SIZE,
    MAX_LATENESS_SEC,
)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
RAW_CSV_PATH      = "data/raw/rbta_ready_ALL.csv"
INJECTED_CSV_PATH = "data/injected/rbta_injected.csv"
AGG_DIR           = "data/aggregated"
FINAL_DIR         = "data/final"
FIGURES_DIR       = "reports/figures"
QUALITY_DIR       = "reports/data_quality"

# Aktifkan True untuk menjalankan attack injection + evaluasi PR-AUC/FNR
USE_INJECTED_DATA = False
INJECTION_SCENARIOS = ["A", "B", "C"]


def safe_step(name: str, fn, *args, **kwargs):
    log.info("=" * 70)
    log.info("  %s", name)
    log.info("=" * 70)
    try:
        result = fn(*args, **kwargs)
        log.info("[OK] %s selesai.", name)
        return result
    except Exception as exc:
        log.error("[ERROR] %s gagal: %s", name, exc)
        traceback.print_exc()
        return None


def main() -> None:
    t_global = time.perf_counter()
    log.info("#" * 70)
    log.info("  RBTA v5 + IF 13-FITUR — FULL PIPELINE ORCHESTRATOR")
    log.info("#" * 70)

    os.makedirs(AGG_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(QUALITY_DIR, exist_ok=True)

    # =========================================================================
    # STEP 1: Preprocessing (selalu dari raw)
    # =========================================================================
    df_raw_original = safe_step(
        "STEP 1 — PREPROCESSING (data raw asli)",
        load_and_prepare,
        RAW_CSV_PATH,
    )
    if df_raw_original is None:
        log.error("Preprocessing gagal. Pipeline dihentikan.")
        return

    # =========================================================================
    # STEP 2: [opsional] Attack Injection
    # =========================================================================
    df_raw_injected = None

    if USE_INJECTED_DATA:
        log.info("STEP 2 — ATTACK INJECTION (scenarios: %s)", INJECTION_SCENARIOS)
        injection_result = safe_step(
            "STEP 2 — ATTACK INJECTION",
            run_injection,
            input_path  = RAW_CSV_PATH,
            output_path = INJECTED_CSV_PATH,
            scenarios   = INJECTION_SCENARIOS,
        )
        if injection_result is not None:
            df_raw_injected = injection_result
            log.info("Dataset injeksi: %d baris (synthetic: %d)",
                     len(df_raw_injected),
                     df_raw_injected["is_synthetic"].sum())
        else:
            log.warning("Injeksi gagal. Lanjut dengan data asli.")
    else:
        log.info("STEP 2 — SKIP (USE_INJECTED_DATA = False)")

    # Dataset yang dipakai pipeline utama
    df_pipeline = df_raw_injected if df_raw_injected is not None else df_raw_original

    # =========================================================================
    # STEP 3: Sensitivity Analysis (cari optimal Δt, enable_adaptive=False)
    # =========================================================================
    log.info("STEP 3 — SENSITIVITY ANALYSIS (enable_adaptive=False)")
    sens_result = safe_step(
        "STEP 3 — SENSITIVITY ANALYSIS",
        sensitivity_analysis,
        df_pipeline,
        DELTA_T_VALUES,
        buffer_size      = BUFFER_SIZE,
        max_lateness_sec = MAX_LATENESS_SEC,
    )

    if sens_result is None:
        log.warning("Sensitivity gagal. Gunakan Δt = 15 sebagai fallback.")
        optimal_dt = 15
        df_sens, meta_map, elastic_map, wmark_map = None, {}, {}, {}
    else:
        df_sens, meta_map, elastic_map, wmark_map = sens_result
        optimal_dt = find_elbow(df_sens)
        print_sensitivity_table(df_sens)
        log.info("Δt optimal = %d menit", optimal_dt)
        safe_step("  PLOT: Sensitivity", plot_sensitivity, df_sens, FIGURES_DIR)

    # =========================================================================
    # STEP 4: RBTA v5 dengan optimal Δt
    # =========================================================================
    log.info("STEP 4 — RBTA v5  (Δt = %d menit)", optimal_dt)
    rbta_result = safe_step(
        "STEP 4 — RBTA v5",
        run_rbta,
        df_pipeline,
        delta_t_minutes    = optimal_dt,
        buffer_size        = BUFFER_SIZE,
        max_lateness_sec   = MAX_LATENESS_SEC,
        enable_adaptive    = False,
    )

    df_meta_rbta = None
    idx_map_rbta = {}
    elastic_rbta = None
    wmark_rbta   = None

    if rbta_result is not None:
        df_meta_rbta, df_compound, idx_map_rbta, elastic_rbta, wmark_rbta = rbta_result

        safe_step("  VALIDATE: Mapping", validate_mapping,
                  df_pipeline, df_meta_rbta, idx_map_rbta)

        if not df_compound.empty:
            p = os.path.join(AGG_DIR, "meta_alerts_compound.csv")
            df_compound.to_csv(p, index=False)
            log.info("Compound (Bucket B) disimpan: %s", p)

        # =====================================================================
        # STEP 5: Feature Engineering f1-f9 → f10-f13
        # =====================================================================
        log.info("STEP 5 — FEATURE ENGINEERING (f1-f13)")
        df_meta_rbta = safe_step("  f1-f9", add_if_features, df_meta_rbta) or df_meta_rbta
        df_meta_rbta = safe_step("  f10-f13", enrich_features, df_meta_rbta) or df_meta_rbta

        # =====================================================================
        # STEP 6: [opsional] Propagasi ground truth label ke meta-alert
        # =====================================================================
        if USE_INJECTED_DATA and df_raw_injected is not None:
            log.info("STEP 6 — PROPAGASI GROUND TRUTH LABEL")
            df_meta_rbta = safe_step(
                "STEP 6 — PROPAGATE LABELS",
                propagate_labels,
                df_meta_rbta,
                df_raw_injected,
                idx_map_rbta,
            ) or df_meta_rbta
        else:
            log.info("STEP 6 — SKIP (USE_INJECTED_DATA = False)")

        p = os.path.join(AGG_DIR, "meta_alerts_rbta.csv")
        df_meta_rbta.to_csv(p, index=False)
        log.info("Meta-alerts RBTA (13 fitur) disimpan: %s", p)
    else:
        log.warning("RBTA gagal. Coba cache sensitivity_analysis.")
        cached = meta_map.get(optimal_dt)
        if cached is not None:
            df_meta_rbta = safe_step("  f1-f9", add_if_features, cached) or cached
            df_meta_rbta = safe_step("  f10-f13", enrich_features, df_meta_rbta) or df_meta_rbta
            p = os.path.join(AGG_DIR, "meta_alerts_rbta.csv")
            df_meta_rbta.to_csv(p, index=False)
            log.info("Meta-alerts (cache, Δt=%d) disimpan: %s", optimal_dt, p)

    # =========================================================================
    # STEP 7: Fixed Window Baseline
    # =========================================================================
    log.info("STEP 7 — FIXED WINDOW BASELINE  (Δt = %d menit)", optimal_dt)
    fixed_result = safe_step(
        "STEP 7 — FIXED WINDOW",
        run_fixed_window,
        df_pipeline,
        delta_t_minutes=optimal_dt,
    )
    df_meta_fixed = None
    if fixed_result is not None:
        df_meta_fixed, _idx_fixed = fixed_result
        if not df_meta_fixed.empty:
            p = os.path.join(AGG_DIR, "meta_alerts_fixed_window.csv")
            df_meta_fixed.to_csv(p, index=False)
            log.info("Fixed Window disimpan: %s", p)

    # =========================================================================
    # STEP 8: Loss Analysis
    # =========================================================================
    if df_meta_fixed is not None and df_meta_rbta is not None:
        safe_step(
            "STEP 8 — LOSS ANALYSIS",
            loss_analysis,
            df_pipeline, df_meta_fixed, df_meta_rbta,
            delta_t_minutes=optimal_dt,
        )

    # =========================================================================
    # STEP 9: Isolation Forest — 13 fitur
    # =========================================================================
    log.info("STEP 9 — ISOLATION FOREST (13 fitur)")
    rbta_csv = os.path.join(AGG_DIR, "meta_alerts_rbta.csv")
    df_scored = None

    if os.path.exists(rbta_csv):
        if_result = safe_step(
            "STEP 9 — ISOLATION FOREST",
            run_isolation_forest,
            rbta_csv,
            FINAL_DIR,
            0.05,   # contamination
            200,    # n_estimators
            "iqr",  # theta_method
            None,   # theta_override
            42,     # random_state
        )
        if if_result is not None:
            df_scored = if_result[0]  # (df_scored, model, scaler, theta)
    else:
        log.warning("File %s tidak ditemukan. Skip IF.", rbta_csv)

    # =========================================================================
    # STEP 10: Evaluasi Komprehensif
    # =========================================================================
    log.info("STEP 10 — EVALUASI KOMPREHENSIF")
    scored_csv = os.path.join(FINAL_DIR, "meta_alerts_scored.csv")

    if df_scored is None and os.path.exists(scored_csv):
        df_scored = pd.read_csv(scored_csv)

    if df_scored is not None:
        safe_step(
            "STEP 10 — COMPREHENSIVE REPORT",
            comprehensive_report,
            df_scored,
            output_dir=QUALITY_DIR,
        )

        # Grafik PR curve dan FNR vs ARR (hanya jika ada ground_truth)
        has_gt = (
            "ground_truth" in df_scored.columns and
            df_scored["ground_truth"].sum() > 0
        )
        if has_gt:
            pr_auc      = compute_pr_auc(df_scored)
            df_tradeoff = compute_fnr_arr_tradeoff(df_scored, df_scored)

            safe_step("  PLOT: PR Curve",
                      plot_pr_curve, df_scored, pr_auc, FIGURES_DIR)
            safe_step("  PLOT: FNR vs ARR",
                      plot_fnr_vs_arr, df_tradeoff, FIGURES_DIR)
        else:
            log.info(
                "Ground truth tidak tersedia. PR-AUC dan FNR vs ARR di-skip. "
                "Set USE_INJECTED_DATA = True dan jalankan ulang."
            )
    else:
        log.warning("df_scored tidak tersedia. Evaluasi di-skip.")

    # =========================================================================
    # STEP 11: Plot detail per Δt (reuse cache sensitivity)
    # =========================================================================
    log.info("STEP 11 — PLOT DETAIL (reuse cache sensitivity)")

    for dt in DELTA_T_VALUES:
        df_dt       = meta_map.get(dt)
        elastic_obj = elastic_map.get(dt)
        wmark_obj   = wmark_map.get(dt)

        if df_dt is not None and not df_dt.empty:
            try:
                plot_severity_dist(df_dt, dt, output_dir=FIGURES_DIR)
                plot_alert_count_dist(df_dt, dt, output_dir=FIGURES_DIR)
            except Exception as exc:
                log.warning("Plot gagal Δt=%d: %s", dt, exc)

        if dt == optimal_dt and elastic_obj is not None and wmark_obj is not None:
            try:
                plot_elastic_dt_history(elastic_obj, dt, output_dir=FIGURES_DIR)
                plot_watermark_stats(wmark_obj, output_dir=FIGURES_DIR)
                print_enhancement_report(wmark_obj, elastic_obj, BUFFER_SIZE)
            except Exception as exc:
                log.warning("Enhancement plot gagal: %s", exc)

    # =========================================================================
    # Cleanup & Summary
    # =========================================================================
    log.info("Membersihkan memori ...")
    for var in (df_raw_original, df_raw_injected, meta_map, elastic_map, wmark_map):
        del var
    gc.collect()

    elapsed = (time.perf_counter() - t_global) * 1000
    log.info("#" * 70)
    log.info("  PIPELINE SELESAI")
    log.info("  Total time     : %.2f menit (%.0f ms)", elapsed / 60000, elapsed)
    log.info("  Δt optimal     : %d menit", optimal_dt)
    log.info("  Injeksi aktif  : %s", USE_INJECTED_DATA)
    log.info("  Output         : %s/ dan %s/", AGG_DIR, FINAL_DIR)
    log.info("  Grafik         : %s/", FIGURES_DIR)
    log.info("  Report         : %s/", QUALITY_DIR)
    log.info("#" * 70)


if __name__ == "__main__":
    main()