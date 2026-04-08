"""
main.py — Orchestrator untuk seluruh pipeline RBTA + Isolation Forest
======================================================================
Urutan eksekusi yang benar:
  1. preprocessing_01       → Load & bersihkan raw CSV
  2. sensitivity_analysis   → Cari optimal_dt DULUAN (ceteris paribus, adaptive=False)
  3. rbta_algorithm_02      → Jalankan RBTA dengan optimal_dt
  4. fixed_window_baseline  → Baseline Fixed Window (Δt sama dengan optimal_dt)
  5. loss_analysis          → Komparasi RBTA vs Fixed Window
  6. isolation_forest       → Anomaly scoring pada meta-alert RBTA optimal
  7. plotting               → Semua grafik (reuse hasil sensitivity_analysis)

Perbaikan dari versi sebelumnya:
  - Optimal_dt dicari DULUAN sebelum menyimpan CSV untuk ML
  - Tidak ada duplikasi komputasi (sensitivity_analysis return cache)
  - Memory management: del + gc.collect() setelah step berat
  - Fault tolerance: try-except per step, pipeline tidak crash total
  - enable_adaptive=False eksplisit di sensitivity_analysis
"""

import sys
import gc
import time
import traceback
from pathlib import Path
from datetime import datetime

# Pastikan root project ada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.preprocessing_01 import load_and_prepare
from src.rbta_algorithm_02 import run_rbta, validate_mapping
from src.fixed_window_baseline import run_fixed_window, loss_analysis
from src.evaluation_03 import (
    sensitivity_analysis,
    plot_sensitivity,
    plot_meta_alert_severity,
    plot_alert_count_per_meta,
    plot_elastic_dt_history,
    plot_watermark_stats,
    print_summary_table,
    print_enhancement_report,
    _find_elbow,
    DELTA_T_VALUES,
    BUFFER_SIZE,
    MAX_LATENESS_SEC,
)
from src.isolation_forest import run_pipeline as run_isolation_forest


# ── Konfigurasi ──────────────────────────────────────────────────────────────
CSV_RAW = "data/rbta_ready_all.csv"

# Auto-create output folder dengan timestamp: output/DDMMYY_HHMMSS/
TIMESTAMP = datetime.now().strftime("%d%m%y_%H%M")
OUTPUT_DIR = f"output/{TIMESTAMP}"


def safe_step(step_name: str, fn, *args, **kwargs):
    """Wrapper try-except per step — pipeline tidak crash total."""
    print(f"\n{'=' * 70}")
    print(f"  {step_name}")
    print("=" * 70)
    try:
        result = fn(*args, **kwargs)
        print(f"[OK] {step_name} selesai.")
        return result
    except Exception as e:
        print(f"\n[ERROR] {step_name} gagal: {e}")
        traceback.print_exc()
        return None


def main():
    t_global = time.perf_counter()

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   RBTA + ISOLATION FOREST — FULL PIPELINE ORCHESTRATOR" + " " * 10 + "#")
    print("#   Dataset: rbta_ready_all.csv (INSTIKI SOC)" + " " * 23 + "#")
    print("#   Alur: optimal_dt dicari DULUAN → baru jalankan ML" + " " * 10 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # =========================================================================
    # STEP 1: Preprocessing
    # =========================================================================
    df_raw = safe_step("STEP 1 — PREPROCESSING", load_and_prepare, CSV_RAW)
    if df_raw is None:
        print("\n[FATAL] Preprocessing gagal. Pipeline dihentikan.")
        return
    print(f"[OK] {len(df_raw):,} baris valid siap diproses.\n")

    # =========================================================================
    # STEP 2: Sensitivity Analysis — CARI OPTIMAL_DT DULUAN
    #            enable_adaptive=False eksplisit (ceteris paribus)
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  STEP 2 — SENSITIVITY ANALYSIS (mencari optimal Δt)")
    print(f"  enable_adaptive = False (statis, ceteris paribus)")
    print(f"{'=' * 70}")

    sens_result = safe_step(
        "STEP 2 — SENSITIVITY ANALYSIS",
        sensitivity_analysis,
        df_raw,
        DELTA_T_VALUES,
        buffer_size      = BUFFER_SIZE,
        max_lateness_sec = MAX_LATENESS_SEC,
    )

    if sens_result is None:
        print("[WARN] Sensitivity analysis gagal. Gunakan Δt = 15 sebagai fallback.")
        optimal_dt = 15
        df_sens, meta_map, elastic_map, wmark_map = None, {}, {}, {}
    else:
        df_sens, meta_map, elastic_map, wmark_map = sens_result
        optimal_dt = _find_elbow(df_sens)
        print(f"\n[INFO] Δt optimal (elbow) = {optimal_dt} menit\n")

        # Plot sensitivity chart
        safe_step("  PLOT: Sensitivity curve", plot_sensitivity, df_sens, OUTPUT_DIR)

    # =========================================================================
    # STEP 3: RBTA dengan optimal_dt — simpan CSV untuk ML
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  STEP 3 — RBTA ENHANCED v3  (Δt = {optimal_dt} menit)")
    print(f"  Menggunakan optimal_dt dari Step 2")
    print(f"{'=' * 70}")

    rbta_result = safe_step(
        "STEP 3 — RBTA",
        run_rbta,
        df_raw,
        delta_t_minutes    = optimal_dt,
        buffer_size        = BUFFER_SIZE,
        max_lateness_sec   = MAX_LATENESS_SEC,
        enable_adaptive    = False,
    )

    df_meta_rbta = None
    if rbta_result is not None:
        df_meta_rbta, idx_map_rbta, elastic_rbta, wmark_rbta = rbta_result

        # Validasi mapping
        safe_step("  VALIDATE: Alert index map", validate_mapping,
                   df_raw, df_meta_rbta, idx_map_rbta)

        # Simpan meta-alerts ke CSV (ini yang akan dipakai ML)
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_out = os.path.join(OUTPUT_DIR, "meta_alerts_rbta.csv")
        df_meta_rbta.to_csv(csv_out, index=False)
        print(f"[OK] Meta-alerts RBTA (Δt={optimal_dt}) disimpan ke: {csv_out}")
    else:
        print("[WARN] RBTA gagal. Coba gunakan cache dari sensitivity_analysis.")
        df_meta_rbta = meta_map.get(optimal_dt)
        if df_meta_rbta is not None:
            import os
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            csv_out = os.path.join(OUTPUT_DIR, "meta_alerts_rbta.csv")
            df_meta_rbta.to_csv(csv_out, index=False)
            print(f"[OK] Meta-alerts RBTA (cache, Δt={optimal_dt}) disimpan ke: {csv_out}")

    # =========================================================================
    # STEP 4: Fixed Window Baseline (Δt sama dengan optimal_dt)
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  STEP 4 — FIXED WINDOW BASELINE  (Δt = {optimal_dt} menit)")
    print(f"{'=' * 70}")

    fixed_result = safe_step(
        "STEP 4 — FIXED WINDOW",
        run_fixed_window,
        df_raw,
        delta_t_minutes=optimal_dt,
    )

    df_meta_fixed = None
    if fixed_result is not None:
        df_meta_fixed, idx_map_fixed = fixed_result
        if not df_meta_fixed.empty:
            import os
            csv_out = os.path.join(OUTPUT_DIR, "meta_alerts_fixed_window.csv")
            df_meta_fixed.to_csv(csv_out, index=False)
            print(f"[OK] Meta-alerts Fixed Window disimpan ke: {csv_out}")

    # =========================================================================
    # STEP 5: Loss Analysis (RBTA vs Fixed Window)
    # =========================================================================
    if df_meta_fixed is not None and df_meta_rbta is not None:
        safe_step(
            "STEP 5 — LOSS ANALYSIS (RBTA vs Fixed Window)",
            loss_analysis,
            df_raw, df_meta_fixed, df_meta_rbta,
            delta_t_minutes=optimal_dt,
        )

    # =========================================================================
    # STEP 6: Plotting detail — REUSE cache, TIDAK re-run RBTA
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  STEP 6 — PLOTTING DETAIL (reuse cache, tanpa re-run)")
    print(f"{'=' * 70}")

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dt in DELTA_T_VALUES:
        df_meta_dt    = meta_map.get(dt)
        elastic_obj   = elastic_map.get(dt)
        wmark_obj     = wmark_map.get(dt)

        if df_meta_dt is not None and not df_meta_dt.empty:
            print(f"  [PLOT] Δt = {dt} menit ...")
            try:
                plot_meta_alert_severity(df_meta_dt, dt, output_dir=OUTPUT_DIR)
                plot_alert_count_per_meta(df_meta_dt, dt, output_dir=OUTPUT_DIR)
            except Exception as e:
                print(f"  [WARN] Plot gagal untuk Δt={dt}: {e}")

        if dt == optimal_dt and elastic_obj and wmark_obj:
            print(f"  [PLOT] Enhancement graphs untuk Δt={optimal_dt} (optimal) ...")
            try:
                plot_elastic_dt_history(elastic_obj, dt, output_dir=OUTPUT_DIR)
                plot_watermark_stats(wmark_obj, output_dir=OUTPUT_DIR)
                print_enhancement_report(wmark_obj, elastic_obj, BUFFER_SIZE)
            except Exception as e:
                print(f"  [WARN] Enhancement plot gagal: {e}")

    # =========================================================================
    # STEP 7: Isolation Forest — PAKAI CSV DARI OPTIMAL_DT
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  STEP 7 — ISOLATION FOREST (meta-alerts Δt = {optimal_dt})")
    print(f"{'=' * 70}")

    rbta_csv = os.path.join(OUTPUT_DIR, "meta_alerts_rbta.csv")
    if os.path.exists(rbta_csv):
        safe_step(
            "STEP 7 — ISOLATION FOREST",
            run_isolation_forest,
            rbta_csv,
            OUTPUT_DIR,
            0.05,   # contamination
            200,    # n_estimators
            "iqr",  # theta_method
            None,   # theta_override
            42,     # random_state
        )
    else:
        print(f"[WARN] File {rbta_csv} tidak ditemukan. Skip Isolation Forest.")

    # =========================================================================
    # Cleanup memory
    # =========================================================================
    print("\n[CLEANUP] Membersihkan memori ...")
    del df_raw
    if 'meta_map' in dir():
        del meta_map, elastic_map, wmark_map
    gc.collect()

    # =========================================================================
    # Ringkasan Akhir
    # =========================================================================
    elapsed_total = (time.perf_counter() - t_global) * 1000
    elapsed_min   = elapsed_total / 60000

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   PIPELINE SELESAI" + " " * 50 + "#")
    print("#" + " " * 68 + "#")
    print(f"#   Total execution time : {elapsed_min:.2f} menit ({elapsed_total:.0f} ms)" + " " * 5 + "#")
    print(f"#   Δt optimal           : {optimal_dt} menit" + " " * (68 - len(f"#   Δt optimal           : {optimal_dt} menit")) + "#")
    print(f"#   Output directory     : {OUTPUT_DIR}/" + " " * (68 - len(f"#   Output directory     : {OUTPUT_DIR}/")) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
