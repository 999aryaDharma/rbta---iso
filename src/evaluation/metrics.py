"""
src/evaluation/metrics.py
==========================
Modul evaluasi tunggal yang menggabungkan:

  [1] Sensitivity Analysis (ARR vs Δt)
      — run_rbta berulang dengan berbagai Δt statis (enable_adaptive=False)
      — elbow detection untuk optimal Δt
      — grafik ARR vs Δt + execution time

  [2] Metrik Akademis (butuh ground_truth dari attack_injector.propagate_labels)
      — PR-AUC  (Precision-Recall Area Under Curve)
      — F0.5    (F-beta Score, β=0.5 — mengutamakan Precision vs Recall)
      — FNR vs ARR  (False Negative Rate vs Alert Reduction Rate)
      — MTTT    (Mean Time to Triage — berapa alert per meta-alert)

  [3] Plotting & Report
      — plot_sensitivity()        → grafik elbow ARR vs Δt
      — plot_pr_curve()           → kurva Precision-Recall
      — plot_fnr_vs_arr()         → tradeoff FNR vs ARR
      — plot_severity_dist()      → distribusi severity meta-alert
      — plot_alert_count_dist()   → distribusi kompresi per meta-alert
      — plot_watermark_stats()    → pie chart late-event handling
      — plot_elastic_dt_history() → riwayat adaptive Δt
      — comprehensive_report()    → teks report lengkap semua metrik

Catatan penggunaan metrik berbasis label:
  Metrik [2] hanya valid jika df_meta memiliki kolom ground_truth (0/1)
  yang diisi oleh attack_injector.propagate_labels(). Jika tidak ada,
  fungsi akan mengembalikan None dan mencetak peringatan.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
)

log = logging.getLogger(__name__)

# ── Palette warna konsisten ───────────────────────────────────────────────────
C_PRIMARY = "#534AB7"
C_ACCENT  = "#1D9E75"
C_WARN    = "#BA7517"
C_RED     = "#C94040"
C_GRAY    = "#888780"

# ── Default parameter sensitivity ────────────────────────────────────────────
DELTA_T_VALUES   = [1, 5, 10, 15, 20, 30, 45, 60]
BUFFER_SIZE      = 50
MAX_LATENESS_SEC = 60.0


# ══════════════════════════════════════════════════════════════════════════════
# [1] SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(
    df_raw:           pd.DataFrame,
    delta_t_values:   list[int] = DELTA_T_VALUES,
    buffer_size:      int       = BUFFER_SIZE,
    max_lateness_sec: float     = MAX_LATENESS_SEC,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    """
    Jalankan RBTA berulang dengan berbagai Δt statis (enable_adaptive=False).

    enable_adaptive DIPAKSA False — sensitivity analysis harus mengukur
    performa parameter statis murni (ceteris paribus).

    Returns
    -------
    df_sens     : DataFrame ringkasan ARR & exec time per Δt
    meta_map    : { delta_t: df_meta }
    elastic_map : { delta_t: ElasticWindow }
    wmark_map   : { delta_t: Watermark }
    """
    from src.engine.rbta_algorithm_02 import run_rbta, ElasticWindow, Watermark

    log.info("[SENSITIVITY] Menjalankan analisis sensitivitas Δt ...")
    n_raw    = len(df_raw)
    results  = []
    meta_map    = {}
    elastic_map = {}
    wmark_map   = {}

    for dt in delta_t_values:
        t0 = time.perf_counter()
        # run_rbta v5 mengembalikan 5 nilai
        df_meta, _df_compound, _idx_map, elastic_obj, wmark_obj = run_rbta(
            df_raw,
            delta_t_minutes  = dt,
            buffer_size      = buffer_size,
            max_lateness_sec = max_lateness_sec,
            enable_adaptive  = False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        n_meta  = len(df_meta)
        arr     = _compute_arr(n_raw, n_meta)

        results.append({
            "delta_t_min":  dt,
            "n_raw":        n_raw,
            "n_meta":       n_meta,
            "arr_pct":      round(arr, 2),
            "exec_time_ms": round(elapsed, 3),
        })
        meta_map[dt]    = df_meta
        elastic_map[dt] = elastic_obj
        wmark_map[dt]   = wmark_obj

        log.info(
            "   Δt=%3d menit  meta=%4d  ARR=%6.2f%%  t=%.1fms",
            dt, n_meta, arr, elapsed,
        )

    df_sens    = pd.DataFrame(results)
    optimal_dt = find_elbow(df_sens)
    log.info("[SENSITIVITY] Selesai. Δt optimal = %d menit", optimal_dt)
    return df_sens, meta_map, elastic_map, wmark_map


def _compute_arr(n_raw: int, n_meta: int) -> float:
    if n_raw == 0:
        return 0.0
    return (1 - n_meta / n_raw) * 100


def find_elbow(df_sens: pd.DataFrame) -> int:
    """Cari titik elbow dengan second derivative + maximum curvature."""
    x = df_sens["delta_t_min"].values
    y = df_sens["arr_pct"].values

    if len(x) < 3:
        return int(x[0])

    d2y = np.diff(np.diff(y))

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    distances = np.abs(y_norm - x_norm) / np.sqrt(2)

    if len(d2y) > 0 and d2y.max() > 0.5:
        elbow_idx = np.argmax(d2y) + 1
    else:
        elbow_idx = np.argmax(distances)

    elbow_idx  = max(0, min(elbow_idx, len(x) - 1))
    optimal_dt = int(x[elbow_idx])
    log.info(
        "  [ELBOW] idx=%d  Δt=%d menit  ARR=%.2f%%",
        elbow_idx, optimal_dt, y[elbow_idx],
    )
    return optimal_dt


# ══════════════════════════════════════════════════════════════════════════════
# [2] METRIK AKADEMIS (butuh kolom ground_truth)
# ══════════════════════════════════════════════════════════════════════════════

def _check_ground_truth(df: pd.DataFrame, caller: str) -> bool:
    """Return True jika kolom ground_truth tersedia dan valid."""
    if "ground_truth" not in df.columns:
        log.warning(
            "[%s] Kolom ground_truth tidak ditemukan. "
            "Jalankan attack_injector.propagate_labels() terlebih dahulu. "
            "Metrik ini di-skip.",
            caller,
        )
        return False
    n_pos = df["ground_truth"].sum()
    if n_pos == 0:
        log.warning(
            "[%s] Tidak ada ground_truth positif (n_pos=0). "
            "Pastikan skenario injeksi dijalankan. Metrik di-skip.",
            caller,
        )
        return False
    return True


def compute_pr_auc(df_scored: pd.DataFrame) -> Optional[float]:
    """
    Hitung Precision-Recall Area Under Curve (PR-AUC).

    Lebih relevan dari ROC-AUC untuk data tidak seimbang (class imbalance).
    ROC-AUC memberikan ilusi performa bagus saat baseline sangat rendah.

    Butuh kolom: ground_truth (0/1), anomaly_score (float [0,1])

    Returns
    -------
    float PR-AUC, atau None jika ground_truth tidak tersedia.
    """
    if not _check_ground_truth(df_scored, "PR-AUC"):
        return None
    if "anomaly_score" not in df_scored.columns:
        log.warning("[PR-AUC] Kolom anomaly_score tidak ditemukan.")
        return None

    y_true  = df_scored["ground_truth"].values
    y_score = df_scored["anomaly_score"].values
    pr_auc  = average_precision_score(y_true, y_score)
    log.info("[PR-AUC] PR-AUC = %.4f", pr_auc)
    return round(float(pr_auc), 4)


def compute_fbeta(
    df_scored: pd.DataFrame,
    beta:      float = 0.5,
    threshold: float = 0.5,
) -> Optional[float]:
    """
    Hitung F-beta Score.

    Default β=0.5 → mengutamakan Precision atas Recall.
    Justifikasi: mengatasi alert fatigue berarti FP harus rendah.
    FP (notifikasi palsu) lebih merusak workflow analis SOC
    daripada FN yang sesekali terlewat.

    Butuh kolom: ground_truth, escalate (atau anomaly_score + threshold)

    Returns
    -------
    float F-beta score, atau None jika ground_truth tidak tersedia.
    """
    if not _check_ground_truth(df_scored, f"F{beta}"):
        return None

    y_true = df_scored["ground_truth"].values

    # Gunakan kolom escalate jika ada, fallback ke threshold anomaly_score
    if "escalate" in df_scored.columns:
        y_pred = df_scored["escalate"].values
    elif "anomaly_score" in df_scored.columns:
        y_pred = (df_scored["anomaly_score"] >= threshold).astype(int).values
    else:
        log.warning("[F%s] Tidak ada kolom escalate atau anomaly_score.", beta)
        return None

    score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
    log.info("[F%.1f] F%.1f-Score = %.4f", beta, beta, score)
    return round(float(score), 4)


def compute_fnr(df_scored: pd.DataFrame) -> Optional[float]:
    """
    Hitung False Negative Rate (FNR).

    FNR = FN / (FN + TP)
        = berapa persen serangan nyata yang ikut terbuang sebagai noise.

    Ini adalah metrik yang paling kritis dari sudut pandang keamanan.
    ARR yang tinggi tidak ada nilainya jika FNR-nya juga tinggi.

    Butuh kolom: ground_truth, escalate

    Returns
    -------
    float FNR [0, 1], atau None jika ground_truth tidak tersedia.
    """
    if not _check_ground_truth(df_scored, "FNR"):
        return None
    if "escalate" not in df_scored.columns:
        log.warning("[FNR] Kolom escalate tidak ditemukan.")
        return None

    y_true = df_scored["ground_truth"].values
    y_pred = df_scored["escalate"].values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    if tp + fn == 0:
        log.warning("[FNR] Tidak ada positif sejati (TP+FN=0).")
        return None

    fnr = fn / (tp + fn)
    log.info("[FNR] FNR = %.4f  (FN=%d  TP=%d)", fnr, fn, tp)
    return round(float(fnr), 4)


def compute_arr_at_threshold(df_scored: pd.DataFrame) -> float:
    """
    Hitung ARR efektif dari hasil decision matrix.

    ARR = (suppress + contextual_anomaly) / total
    """
    n_total   = len(df_scored)
    n_suppress = (df_scored.get("action", pd.Series()) == "SUPPRESS").sum()
    if n_total == 0:
        return 0.0
    return round(float(n_suppress / n_total * 100), 2)


def compute_mttt(df_scored: pd.DataFrame) -> dict:
    """
    Hitung Mean Time to Triage (MTTT) — proxy kognitif.

    MTTT di sini diukur sebagai rata-rata jumlah raw alert per meta-alert
    yang perlu ditriase analis (hanya yang ESCALATE).

    Semakin tinggi nilai ini, semakin banyak "pra-triage" yang sudah
    dilakukan sistem, sehingga analis melihat insiden bukan alert individu.

    Butuh kolom: alert_count, escalate (atau action)

    Returns
    -------
    dict dengan key:
      avg_alerts_per_meta_all      — rata-rata alert per meta-alert (semua)
      avg_alerts_per_meta_escalate — rata-rata alert per meta-alert yang dieskalasi
      compression_ratio            — total raw / total meta (seluruh dataset)
      triage_load_reduction_pct    — penurunan beban triage vs tanpa agregasi
    """
    if "alert_count" not in df_scored.columns:
        log.warning("[MTTT] Kolom alert_count tidak ditemukan.")
        return {}

    n_meta    = len(df_scored)
    n_raw_est = df_scored["alert_count"].sum()

    avg_all = df_scored["alert_count"].mean()

    if "escalate" in df_scored.columns:
        escalated = df_scored[df_scored["escalate"] == 1]
        avg_esc   = escalated["alert_count"].mean() if len(escalated) > 0 else 0.0
    else:
        avg_esc   = avg_all

    compression = n_raw_est / n_meta if n_meta > 0 else 1.0
    load_reduce = (1 - n_meta / n_raw_est) * 100 if n_raw_est > 0 else 0.0

    result = {
        "avg_alerts_per_meta_all":      round(float(avg_all), 2),
        "avg_alerts_per_meta_escalate": round(float(avg_esc), 2),
        "compression_ratio":            round(float(compression), 2),
        "triage_load_reduction_pct":    round(float(load_reduce), 2),
    }
    log.info(
        "[MTTT] avg_all=%.1f  avg_escalate=%.1f  compression=%.1fx  "
        "load_reduction=%.1f%%",
        avg_all, avg_esc, compression, load_reduce,
    )
    return result


def compute_fnr_arr_tradeoff(
    df_meta_raw: pd.DataFrame,
    df_scored:   pd.DataFrame,
    thresholds:  Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Hitung pasangan (ARR, FNR) di berbagai threshold anomaly_score.

    Digunakan untuk membuat grafik korelasi ARR vs FNR yang diminta
    dosen penguji. Grafik ini membuktikan bahwa sistem tidak
    mengorbankan FNR demi ARR yang tinggi.

    Butuh kolom: ground_truth, anomaly_score di df_scored.

    Returns
    -------
    pd.DataFrame dengan kolom: threshold, arr_pct, fnr, precision, recall
    """
    if not _check_ground_truth(df_scored, "FNR-ARR tradeoff"):
        return pd.DataFrame()
    if "anomaly_score" not in df_scored.columns:
        log.warning("[FNR-ARR] Kolom anomaly_score tidak ditemukan.")
        return pd.DataFrame()

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 41).tolist()

    y_true  = df_scored["ground_truth"].values
    y_score = df_scored["anomaly_score"].values
    n_total = len(df_scored)

    rows = []
    for thr in thresholds:
        y_pred     = (y_score >= thr).astype(int)
        tp         = int(((y_true == 1) & (y_pred == 1)).sum())
        fn         = int(((y_true == 1) & (y_pred == 0)).sum())
        fp         = int(((y_true == 0) & (y_pred == 1)).sum())
        tn         = int(((y_true == 0) & (y_pred == 0)).sum())
        n_suppress = int((y_pred == 0).sum())

        fnr       = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        arr       = n_suppress / n_total * 100 if n_total > 0 else 0.0

        rows.append({
            "threshold": round(thr, 3),
            "arr_pct":   round(arr, 2),
            "fnr":       round(fnr, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# [3] PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("[PLOT] Tersimpan: %s", path)
    plt.close(fig)


def plot_sensitivity(
    df_sens:    pd.DataFrame,
    output_dir: str = "reports/figures",
) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        df_sens["delta_t_min"], df_sens["arr_pct"],
        color=C_PRIMARY, marker="o", linewidth=2.2, markersize=7,
        label="Alert Reduction Rate (%)", zorder=3,
    )
    ax1.fill_between(df_sens["delta_t_min"], df_sens["arr_pct"],
                     alpha=0.08, color=C_PRIMARY)

    for _, row in df_sens.iterrows():
        ax1.annotate(
            f"{row['arr_pct']:.1f}%",
            xy=(row["delta_t_min"], row["arr_pct"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8.5, color=C_PRIMARY,
        )

    ax1.set_xlabel("Time-Window Δt (menit)", fontsize=11)
    ax1.set_ylabel("Alert Reduction Rate (%)", fontsize=11, color=C_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.set_xticks(df_sens["delta_t_min"])

    ax2 = ax1.twinx()
    ax2.plot(
        df_sens["delta_t_min"], df_sens["exec_time_ms"],
        color=C_ACCENT, marker="s", linewidth=1.5, markersize=5,
        linestyle="--", label="Execution Time (ms)", zorder=2,
    )
    ax2.set_ylabel("Execution Time (ms)", fontsize=11, color=C_ACCENT)
    ax2.tick_params(axis="y", labelcolor=C_ACCENT)

    optimal_dt  = find_elbow(df_sens)
    elbow_row   = df_sens[df_sens["delta_t_min"] == optimal_dt].iloc[0]
    ax1.axvline(optimal_dt, color=C_WARN, linewidth=1.4,
                linestyle=":", label=f"Elbow Δt={optimal_dt}m")
    ax1.annotate(
        f"Δt optimal\n= {optimal_dt} menit",
        xy=(optimal_dt, elbow_row["arr_pct"]),
        xytext=(optimal_dt + 3, elbow_row["arr_pct"] - 10),
        fontsize=9, color=C_WARN,
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.2),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
               fontsize=9, framealpha=0.9)

    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title(
        "Analisis Sensitivitas: ARR vs Time-Window (Δt)\n"
        "Rule-Based Temporal Aggregation Enhanced — Data SIEM Wazuh INSTIKI",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    _savefig(fig, f"{output_dir}/sensitivity_ARR_vs_delta_t.png")


def plot_pr_curve(
    df_scored:  pd.DataFrame,
    pr_auc:     Optional[float],
    output_dir: str = "reports/figures",
) -> None:
    if not _check_ground_truth(df_scored, "plot_pr_curve"):
        return
    if "anomaly_score" not in df_scored.columns:
        return

    y_true  = df_scored["ground_truth"].values
    y_score = df_scored["anomaly_score"].values
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=C_PRIMARY, linewidth=2.2,
            label=f"PR curve (AUC = {pr_auc:.4f})" if pr_auc else "PR curve")
    ax.fill_between(recall, precision, alpha=0.08, color=C_PRIMARY)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_title(
        "Precision-Recall Curve — Isolation Forest 13 Fitur\n"
        "Data SIEM Wazuh INSTIKI (dengan synthetic attack injection)",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    _savefig(fig, f"{output_dir}/pr_curve.png")


def plot_fnr_vs_arr(
    df_tradeoff: pd.DataFrame,
    output_dir:  str = "reports/figures",
) -> None:
    if df_tradeoff.empty:
        log.warning("[plot_fnr_vs_arr] DataFrame kosong — skip.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        df_tradeoff["arr_pct"], df_tradeoff["fnr"],
        c=df_tradeoff["threshold"], cmap="viridis_r",
        s=40, zorder=3, alpha=0.85,
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Threshold θ", fontsize=9)

    ax.set_xlabel("Alert Reduction Rate / ARR (%)", fontsize=11)
    ax.set_ylabel("False Negative Rate / FNR", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.02, 1.05)

    # Garis referensi FNR = 0.1 (batas aman industri)
    ax.axhline(0.10, color=C_RED, linewidth=1.2, linestyle="--",
               label="FNR = 0.10 (batas aman)")
    ax.legend(fontsize=9)

    ax.grid(linestyle="--", alpha=0.35)
    ax.set_title(
        "Tradeoff ARR vs FNR — Isolasi Forest\n"
        "Setiap titik = satu nilai threshold θ",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    _savefig(fig, f"{output_dir}/fnr_vs_arr_tradeoff.png")


def plot_severity_dist(
    df_meta:    pd.DataFrame,
    delta_t:    int,
    output_dir: str = "reports/figures",
) -> None:
    if df_meta.empty:
        return
    counts = df_meta["max_severity"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=C_PRIMARY, alpha=0.82, width=0.6, zorder=3)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Max Severity (rule.level)", fontsize=11)
    ax.set_ylabel("Jumlah Meta-Alert", fontsize=11)
    ax.set_title(f"Distribusi Severity Meta-Alert  (Δt = {delta_t} menit)",
                 fontsize=11, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    _savefig(fig, f"{output_dir}/meta_alert_severity_dt{delta_t}.png")


def plot_alert_count_dist(
    df_meta:    pd.DataFrame,
    delta_t:    int,
    output_dir: str = "reports/figures",
) -> None:
    if df_meta.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_meta["alert_count"], bins=20, color=C_ACCENT, alpha=0.8, zorder=3)
    ax.set_xlabel("Jumlah Raw Alert per Meta-Alert", fontsize=11)
    ax.set_ylabel("Frekuensi", fontsize=11)
    ax.set_title(f"Distribusi Kompresi Alert per Insiden  (Δt = {delta_t} menit)",
                 fontsize=11, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    _savefig(fig, f"{output_dir}/alert_count_distribution_dt{delta_t}.png")


def plot_watermark_stats(wmark, output_dir: str = "reports/figures") -> None:
    labels      = ["On-time", "Late (accepted)", "Late (dropped)"]
    sizes       = [wmark.n_on_time, wmark.n_late_ok, wmark.n_late_drop]
    colors      = [C_PRIMARY, C_ACCENT, C_RED]
    sizes_clean = [max(s, 0) for s in sizes]
    if sum(sizes_clean) == 0:
        log.warning("[plot_watermark] Stats kosong — skip.")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    _, _, autotexts = ax.pie(
        sizes_clean, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 0.1 else "",
        startangle=140, pctdistance=0.80,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(
        f"Klasifikasi Event oleh Watermark\n"
        f"(max_lateness = {wmark.max_lateness.total_seconds():.0f} detik)",
        fontsize=11, pad=10,
    )
    fig.tight_layout()
    _savefig(fig, f"{output_dir}/watermark_event_classification.png")


def plot_elastic_dt_history(
    elastic,
    delta_t_base: int,
    output_dir:   str = "reports/figures",
) -> None:
    if not elastic or not elastic.dt_history:
        return
    timestamps = [t for t, _ in elastic.dt_history]
    dt_vals    = [v for _, v in elastic.dt_history]
    fig, ax    = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, dt_vals, color=C_PRIMARY, linewidth=1.5, alpha=0.85)
    ax.axhline(delta_t_base, color=C_WARN, linewidth=1.2,
               linestyle="--", label=f"Δt base = {delta_t_base} menit")
    ax.axhline(elastic.min_dt.total_seconds() / 60, color=C_RED,
               linewidth=0.9, linestyle=":", label="Δt min")
    ax.axhline(elastic.max_dt.total_seconds() / 60, color=C_ACCENT,
               linewidth=0.9, linestyle=":", label="Δt max")
    ax.set_xlabel("Timestamp Event", fontsize=11)
    ax.set_ylabel("Δt Adaptif (menit)", fontsize=11)
    ax.set_title(
        f"Riwayat Adaptive Δt (ElasticWindow)  —  Δt base = {delta_t_base} menit",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()
    _savefig(fig, f"{output_dir}/elastic_dt_history_dt{delta_t_base}.png")


# ══════════════════════════════════════════════════════════════════════════════
# [4] COMPREHENSIVE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_sensitivity_table(df_sens: pd.DataFrame) -> None:
    log.info("\n" + "=" * 65)
    log.info("  TABEL SENSITIVITAS Δt  [buffer k=%d | lateness=%.0fs]",
             BUFFER_SIZE, MAX_LATENESS_SEC)
    log.info("=" * 65)
    log.info("  %-12s %-12s %-12s %-10s %-10s",
             "Δt (menit)", "Raw Alert", "Meta-Alert", "ARR (%)", "Exec (ms)")
    log.info("  " + "-" * 58)
    for _, row in df_sens.iterrows():
        log.info("  %-12d %-12d %-12d %-10.2f %-10.3f",
                 int(row["delta_t_min"]), int(row["n_raw"]),
                 int(row["n_meta"]), row["arr_pct"], row["exec_time_ms"])
    log.info("=" * 65)


def print_enhancement_report(wmark, elastic, buffer_size: int) -> None:
    total = wmark.n_on_time + wmark.n_late_ok + wmark.n_late_drop
    log.info("\n" + "=" * 65)
    log.info("  LAPORAN ENHANCEMENT — RBTA v5")
    log.info("=" * 65)
    log.info("  [1] Out-of-Order Buffer (Min-Heap)")
    log.info("      Buffer size k  = %d", buffer_size)
    log.info("      Kompleksitas   = O(n log %d)", buffer_size)
    log.info("  [2] Elastic Time-Window")
    if elastic and elastic.dt_history:
        dt_vals = [v for _, v in elastic.dt_history]
        adapted = abs(elastic.current_minutes - elastic.base_dt.total_seconds() / 60) > 0.1
        log.info("      Δt akhir sesi  = %.2f menit", elastic.current_minutes)
        log.info("      Δt min/max     = %.2f / %.2f menit",
                 min(dt_vals), max(dt_vals))
        log.info("      Adaptasi       = %s", "Ya" if adapted else "Tidak")
    else:
        log.info("      (nonaktif atau belum cukup warmup data)")
    log.info("  [3] Watermark Late-Event Handling")
    log.info("      Max lateness   = %.0f detik", wmark.max_lateness.total_seconds())
    if total > 0:
        log.info("      On-time        = %d", wmark.n_on_time)
        log.info("      Late accepted  = %d (%.1f%%)",
                 wmark.n_late_ok, wmark.n_late_ok / total * 100)
        log.info("      Late dropped   = %d (%.1f%%)",
                 wmark.n_late_drop, wmark.n_late_drop / total * 100)
    log.info("=" * 65)


def comprehensive_report(
    df_scored:   pd.DataFrame,
    output_dir:  str = "reports/data_quality",
) -> str:
    """
    Buat teks report lengkap semua metrik evaluasi.

    Jika ground_truth tidak tersedia, hanya ARR dan MTTT yang dihitung.
    Jika tersedia, semua metrik dihitung.

    Returns
    -------
    str — teks report yang juga disimpan ke file
    """
    has_gt = "ground_truth" in df_scored.columns and df_scored["ground_truth"].sum() > 0

    arr_eff = compute_arr_at_threshold(df_scored)
    mttt    = compute_mttt(df_scored)

    pr_auc  = compute_pr_auc(df_scored)   if has_gt else None
    f05     = compute_fbeta(df_scored, beta=0.5) if has_gt else None
    f1      = compute_fbeta(df_scored, beta=1.0) if has_gt else None
    fnr     = compute_fnr(df_scored)      if has_gt else None

    lines = [
        "",
        "=" * 70,
        "  LAPORAN EVALUASI KOMPREHENSIF — RBTA + IF 13-FITUR",
        "  INSTIKI SOC — Skripsi Alert Fatigue",
        "=" * 70,
        "",
        "  [A] ALERT REDUCTION RATE (ARR)",
        f"      ARR efektif (suppress+contextual) : {arr_eff:.2f}%",
        "",
        "  [B] MEAN TIME TO TRIAGE (MTTT)",
        f"      Avg alert per meta-alert (semua)  : {mttt.get('avg_alerts_per_meta_all', 0):.1f}",
        f"      Avg alert per meta-alert (escalate): {mttt.get('avg_alerts_per_meta_escalate', 0):.1f}",
        f"      Kompresi (raw/meta)                : {mttt.get('compression_ratio', 1):.1f}x",
        f"      Reduksi beban triage               : {mttt.get('triage_load_reduction_pct', 0):.1f}%",
        "",
    ]

    if has_gt:
        lines += [
            "  [C] METRIK BERBASIS GROUND TRUTH (synthetic injection)",
            f"      PR-AUC                           : {pr_auc:.4f}" if pr_auc else "      PR-AUC : N/A",
            f"      F1-Score  (β=1.0)                : {f1:.4f}"    if f1    else "      F1     : N/A",
            f"      F0.5-Score (β=0.5, Precision++)  : {f05:.4f}"   if f05   else "      F0.5   : N/A",
            f"      False Negative Rate (FNR)        : {fnr:.4f}"   if fnr   else "      FNR    : N/A",
            "",
            "  Catatan F0.5: β=0.5 mengutamakan Precision atas Recall.",
            "  Dalam konteks alert fatigue, FP lebih merusak workflow SOC",
            "  dibandingkan FN sesekali. Lihat Bab 4 untuk justifikasi.",
            "",
            "  [D] DISTRIBUSI DECISION",
        ]
        if "decision" in df_scored.columns:
            for dec, cnt in df_scored["decision"].value_counts().items():
                pct = cnt / len(df_scored) * 100
                lines.append(f"      {dec:<25}: {cnt:>4}  ({pct:.1f}%)")
    else:
        lines += [
            "  [C] METRIK BERBASIS GROUND TRUTH",
            "      Tidak tersedia — jalankan attack_injector.py terlebih dahulu",
            "      untuk mendapatkan label ground truth yang valid.",
        ]

    lines.append("=" * 70)
    report = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("[REPORT] Disimpan: %s", report_path)
    log.info(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point — jalankan evaluasi pada CSV yang sudah diskor
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluasi metrik untuk meta_alerts_scored.csv"
    )
    parser.add_argument(
        "--input", default="data/final/meta_alerts_scored.csv",
        help="Path ke meta_alerts_scored.csv",
    )
    parser.add_argument(
        "--output-dir", default="reports",
        help="Direktori output untuk grafik dan report",
    )
    args = parser.parse_args()

    log.info("Loading: %s", args.input)
    df = pd.read_csv(args.input)

    comprehensive_report(df, output_dir=f"{args.output_dir}/data_quality")

    # Grafik tradeoff FNR vs ARR (hanya jika ada ground_truth)
    if "ground_truth" in df.columns and df["ground_truth"].sum() > 0:
        df_tradeoff = compute_fnr_arr_tradeoff(df, df)
        plot_pr_curve(df, compute_pr_auc(df), output_dir=f"{args.output_dir}/figures")
        plot_fnr_vs_arr(df_tradeoff, output_dir=f"{args.output_dir}/figures")

    log.info("Evaluasi selesai.")