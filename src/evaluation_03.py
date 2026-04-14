"""
evaluation_03.py  —  Adaptasi untuk rbta_ready_all.csv
==========================================================
Perubahan dari versi sebelumnya:

  [BUG-FIX] return_internals tidak ada di signature run_rbta()
      Versi lama memanggil run_rbta(..., return_internals=True) dan
      mencoba unpack 3 nilai (df_meta, elastic_obj, wmark_obj).
      run_rbta() selalu mengembalikan 4 nilai: (df_meta, idx_map,
      elastic, wmark). Parameter return_internals tidak pernah ada.
      Fix: hapus return_internals, unpack 4 nilai dengan benar.

  [ADAPT] path default ke rbta_ready_all.csv
      Semua referensi data default diubah ke data/rbta_ready_all.csv.

  [ADAPT] MAX_LATENESS_SEC disesuaikan
      Dataset stratified_sample mencakup 184 hari. Out-of-order yang
      mungkin terjadi adalah dari artifact CSV export, bukan dari
      network delay aktual. Nilai 60 detik lebih tepat dari 600 detik
      sebelumnya (600 detik terlalu permisif untuk batch CSV).

Metrik:
  - Alert Reduction Rate (ARR)  → metrik utama
  - Execution Time              → kinerja O(n log k)
  - Sensitivity Analysis        → ARR vs Δt (grafik elbow)
  - Severity distribution       → distribusi max_severity per meta-alert
  - Watermark report            → statistik late event handling
  - Elastic Δt range            → rentang adaptasi window
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.rbta_core import run_rbta, OutOfOrderBuffer, ElasticWindow, Watermark


DELTA_T_VALUES    = [1, 5, 10, 15, 20, 30, 45, 60]
BUFFER_SIZE       = 50
MAX_LATENESS_SEC  = 60.0    # [ADAPT] 60 detik — lebih tepat untuk batch CSV
ENABLE_ADAPTIVE   = True

COLOR_PRIMARY = "#534AB7"
COLOR_ACCENT  = "#1D9E75"
COLOR_WARN    = "#BA7517"
COLOR_GRAY    = "#888780"
COLOR_LATE    = "#C94040"


def compute_arr(n_raw: int, n_meta: int) -> float:
    if n_raw == 0:
        return 0.0
    return (1 - n_meta / n_raw) * 100


def sensitivity_analysis(
    df_raw:           pd.DataFrame,
    delta_t_values:   list[int] = DELTA_T_VALUES,
    buffer_size:      int       = BUFFER_SIZE,
    max_lateness_sec: float     = MAX_LATENESS_SEC,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, ElasticWindow], dict[int, Watermark]]:
    """
    Jalankan RBTA berulang kali dengan berbagai nilai Δt statis.

    PENTING: enable_adaptive DIPAKSA False agar Δt tidak berubah-ubah
    di latar belakang. Analisis sensitivitas harus mengukur performa
    parameter statis murni (ceteris paribus).

    Returns
    -------
    df_sens      : DataFrame ringkasan ARR & exec time per Δt.
    meta_map     : dict { delta_t : df_meta } — hasil RBTA per Δt.
    elastic_map  : dict { delta_t : ElasticWindow }.
    wmark_map    : dict { delta_t : Watermark }.
    """
    print("\n[SENSITIVITY] Menjalankan analisis sensitivitas statis Δt...")
    n_raw   = len(df_raw)
    results  = []
    meta_map = {}
    elastic_map  = {}
    wmark_map    = {}

    for dt in delta_t_values:
        t0 = time.perf_counter()

        # FIX: Paksa enable_adaptive=False agar Δt tidak di-override oleh ElasticWindow
        # [FIX-C] unpack 5 nilai: (df_meta, df_compound, idx_map, elastic, wmark)
        df_meta, _, idx_map, elastic_obj, wmark_obj = run_rbta(
            df_raw,
            delta_t_minutes  = dt,
            buffer_size      = buffer_size,
            max_lateness_sec = max_lateness_sec,
            enable_adaptive  = False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        n_meta = len(df_meta)
        arr    = compute_arr(n_raw, n_meta)

        results.append({
            "delta_t_min":  dt,
            "n_raw":        n_raw,
            "n_meta":       n_meta,
            "arr_pct":      round(arr, 2),
            "exec_time_ms": round(elapsed, 3),
        })

        # Simpan hasil per-Δt untuk reuse di plotting & ML
        meta_map[dt]     = df_meta
        elastic_map[dt]  = elastic_obj
        wmark_map[dt]    = wmark_obj

        print(f"   Δt={dt:>3} menit  →  meta={n_meta:>4}  ARR={arr:>6.2f}%  t={elapsed:.2f}ms")

    df_sens    = pd.DataFrame(results)
    optimal_dt = _find_elbow(df_sens)
    print(f"\n[OK] Analisis selesai. Titik optimal (elbow) ada di Δt = {optimal_dt} menit\n")
    return df_sens, meta_map, elastic_map, wmark_map


def _find_elbow(df_sens: pd.DataFrame) -> int:
    """
    Cari titik siku (elbow) menggunakan second derivative dan
    maximum curvature (jarak dari garis diagonal normalisasi).
    """
    x = df_sens["delta_t_min"].values
    y = df_sens["arr_pct"].values

    if len(x) < 3:
        return int(x[0])

    dy  = np.diff(y)
    d2y = np.diff(dy)

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    distances = np.abs(y_norm - x_norm) / np.sqrt(2)

    if len(d2y) > 0 and d2y.max() > 0.5:
        elbow_idx = np.argmax(d2y) + 1
    else:
        elbow_idx = np.argmax(distances)

    elbow_idx  = max(0, min(elbow_idx, len(x) - 1))
    optimal_dt = int(x[elbow_idx])

    print(f"  [ELBOW] idx={elbow_idx}, Δt={optimal_dt}m, ARR={y[elbow_idx]:.2f}%")
    return optimal_dt


def plot_sensitivity(df_sens: pd.DataFrame, output_dir: str = "output") -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(df_sens["delta_t_min"], df_sens["arr_pct"],
             color=COLOR_PRIMARY, marker="o", linewidth=2.2,
             markersize=7, label="Alert Reduction Rate (%)", zorder=3)
    ax1.fill_between(df_sens["delta_t_min"], df_sens["arr_pct"],
                     alpha=0.08, color=COLOR_PRIMARY)

    for _, row in df_sens.iterrows():
        ax1.annotate(f"{row['arr_pct']:.1f}%",
                     xy=(row["delta_t_min"], row["arr_pct"]),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=8.5, color=COLOR_PRIMARY)

    ax1.set_xlabel("Time-Window Δt (menit)", fontsize=11)
    ax1.set_ylabel("Alert Reduction Rate / ARR (%)", fontsize=11, color=COLOR_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=COLOR_PRIMARY)
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.set_xticks(df_sens["delta_t_min"])

    ax2 = ax1.twinx()
    ax2.plot(df_sens["delta_t_min"], df_sens["exec_time_ms"],
             color=COLOR_ACCENT, marker="s", linewidth=1.5,
             markersize=5, linestyle="--", label="Execution Time (ms)", zorder=2)
    ax2.set_ylabel("Execution Time (ms)", fontsize=11, color=COLOR_ACCENT)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACCENT)

    elbow_dt  = _find_elbow(df_sens)
    elbow_row = df_sens[df_sens["delta_t_min"] == elbow_dt].iloc[0]
    ax1.axvline(elbow_dt, color=COLOR_WARN, linewidth=1.4,
                linestyle=":", label=f"Elbow point (Δt={elbow_dt}m)")
    ax1.annotate(f"Δt optimal\n= {elbow_dt} menit",
                 xy=(elbow_dt, elbow_row["arr_pct"]),
                 xytext=(elbow_dt + 3, elbow_row["arr_pct"] - 10),
                 fontsize=9, color=COLOR_WARN,
                 arrowprops=dict(arrowstyle="->", color=COLOR_WARN, lw=1.2))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9, framealpha=0.9)

    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title(
        "Analisis Sensitivitas: ARR vs Time-Window (Δt)\n"
        "Rule-Based Temporal Aggregation Enhanced — Data SIEM Wazuh INSTIKI",
        fontsize=11, pad=12,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/sensitivity_ARR_vs_delta_t.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Grafik tersimpan: {out_path}")
    plt.close()


def plot_elastic_dt_history(
    elastic:       ElasticWindow,
    delta_t_base:  int,
    output_dir:    str = "output",
) -> None:
    if not elastic or not elastic.dt_history:
        return

    timestamps = [t for t, _ in elastic.dt_history]
    dt_vals    = [v for _, v in elastic.dt_history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, dt_vals, color=COLOR_PRIMARY, linewidth=1.5, alpha=0.85)
    ax.axhline(delta_t_base, color=COLOR_WARN, linewidth=1.2,
               linestyle="--", label=f"Δt base = {delta_t_base} menit")
    ax.axhline(elastic.min_dt.total_seconds() / 60, color=COLOR_LATE,
               linewidth=0.9, linestyle=":", label="Δt min")
    ax.axhline(elastic.max_dt.total_seconds() / 60, color=COLOR_ACCENT,
               linewidth=0.9, linestyle=":", label="Δt max")
    ax.set_xlabel("Timestamp Event", fontsize=11)
    ax.set_ylabel("Δt Adaptif (menit)", fontsize=11)
    ax.set_title(
        f"Riwayat Adaptive Δt (ElasticWindow)  —  Δt base = {delta_t_base} menit\n"
        "Δt menyusut saat burst, melebar saat frekuensi rendah",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/elastic_dt_history_dt{delta_t_base}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Riwayat Elastic Δt tersimpan: {out_path}")
    plt.close()


def plot_watermark_stats(wmark: Watermark, output_dir: str = "output") -> None:
    labels       = ["On-time", "Late (accepted)", "Late (dropped)"]
    sizes        = [wmark.n_on_time, wmark.n_late_ok, wmark.n_late_drop]
    colors       = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_LATE]
    sizes_clean  = [max(s, 0) for s in sizes]
    if sum(sizes_clean) == 0:
        print("[WARN] Watermark stats kosong — skip pie chart")
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

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/watermark_event_classification.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Watermark pie chart tersimpan: {out_path}")
    plt.close()


def plot_meta_alert_severity(
    df_meta:    pd.DataFrame,
    delta_t:    int,
    output_dir: str = "output",
) -> None:
    if df_meta.empty:
        return
    counts = df_meta["max_severity"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=COLOR_PRIMARY, alpha=0.82, width=0.6, zorder=3)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Max Severity (rule.level)", fontsize=11)
    ax.set_ylabel("Jumlah Meta-Alert", fontsize=11)
    ax.set_title(f"Distribusi Tingkat Keparahan Meta-Alert  (Δt = {delta_t} menit)",
                 fontsize=11, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    out_path = f"{output_dir}/meta_alert_severity_dt{delta_t}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Distribusi severity tersimpan: {out_path}")
    plt.close()


def plot_alert_count_per_meta(
    df_meta:    pd.DataFrame,
    delta_t:    int,
    output_dir: str = "output",
) -> None:
    if df_meta.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_meta["alert_count"], bins=20, color=COLOR_ACCENT, alpha=0.8, zorder=3)
    ax.set_xlabel("Jumlah Raw Alert per Meta-Alert", fontsize=11)
    ax.set_ylabel("Frekuensi", fontsize=11)
    ax.set_title(f"Distribusi Kompresi Alert per Insiden  (Δt = {delta_t} menit)",
                 fontsize=11, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path = f"{output_dir}/alert_count_distribution_dt{delta_t}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Histogram kompresi tersimpan: {out_path}")
    plt.close()


def print_summary_table(df_sens: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  TABEL HASIL EKSPERIMEN — ANALISIS SENSITIVITAS Δt")
    print(f"  [RBTA Enhanced v3 | buffer k={BUFFER_SIZE} | lateness={MAX_LATENESS_SEC}s]")
    print("=" * 65)
    print(f"  {'Δt (menit)':<12} {'Raw Alert':<12} {'Meta-Alert':<12} "
          f"{'ARR (%)':<10} {'Exec (ms)':<10}")
    print("-" * 65)
    for _, row in df_sens.iterrows():
        print(f"  {int(row['delta_t_min']):<12} {int(row['n_raw']):<12} "
              f"{int(row['n_meta']):<12} {row['arr_pct']:<10.2f} "
              f"{row['exec_time_ms']:<10.3f}")
    print("=" * 65 + "\n")


def print_enhancement_report(
    wmark:       Watermark,
    elastic:     ElasticWindow | None,
    buffer_size: int,
) -> None:
    total = wmark.n_on_time + wmark.n_late_ok + wmark.n_late_drop
    print("\n" + "=" * 65)
    print("  LAPORAN ENHANCEMENT AKADEMIS — RBTA v3")
    print("=" * 65)
    print(f"  [1] Out-of-Order Buffer (Min-Heap)")
    print(f"      Buffer size (k)    = {buffer_size}")
    print(f"      Kompleksitas       = O(n log {buffer_size}) — near-linear")
    print(f"  [2] Elastic Time-Window (Adaptive Δt)")
    if elastic and elastic.dt_history:
        dt_vals = [v for _, v in elastic.dt_history]
        print(f"      Δt akhir sesi      = {elastic.current_minutes:.2f} menit")
        print(f"      Δt min/max dicapai = {min(dt_vals):.2f} / {max(dt_vals):.2f} menit")
        adapted = "Ya" if abs(elastic.current_minutes - elastic.base_dt.total_seconds()/60) > 0.1 else "Tidak"
        print(f"      Adaptasi terjadi   = {adapted}")
    else:
        print(f"      (nonaktif atau belum cukup data warmup)")
    print(f"  [3] Watermark Late-Event Handling")
    print(f"      Max lateness       = {wmark.max_lateness.total_seconds():.0f} detik")
    if total > 0:
        print(f"      On-time events     = {wmark.n_on_time:,}")
        print(f"      Late (accepted)    = {wmark.n_late_ok:,}  ({wmark.n_late_ok/total*100:.1f}%)")
        print(f"      Late (dropped)     = {wmark.n_late_drop:,}  ({wmark.n_late_drop/total*100:.1f}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    from etl.preprocessing_01 import load_and_prepare

    # [ADAPT] default ke rbta_ready_all.csv
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/rbta_ready_all.csv"
    df_raw   = load_and_prepare(csv_path)

    df_sens, meta_map, elastic_map, wmark_map = sensitivity_analysis(
        df_raw, DELTA_T_VALUES,
        buffer_size      = BUFFER_SIZE,
        max_lateness_sec = MAX_LATENESS_SEC,
    )
    print_summary_table(df_sens)
    plot_sensitivity(df_sens)

    OPTIMAL_DT = _find_elbow(df_sens)
    print(f"\n[INFO] Δt optimal (elbow) = {OPTIMAL_DT} menit\n")

    os.makedirs("output", exist_ok=True)

    # Gunakan hasil yang sudah tersimpan — tidak perlu re-run RBTA
    for dt in DELTA_T_VALUES:
        df_meta_dt    = meta_map.get(dt)
        elastic_obj   = elastic_map.get(dt)
        wmark_obj     = wmark_map.get(dt)

        if df_meta_dt is not None and not df_meta_dt.empty:
            print(f"[DETAIL] Plot detail untuk Δt = {dt} menit...")
            plot_meta_alert_severity(df_meta_dt, dt)
            plot_alert_count_per_meta(df_meta_dt, dt)

        if dt == OPTIMAL_DT and elastic_obj and wmark_obj:
            print(f"[INFO] Δt {dt} menit adalah optimal — mencetak grafik enhancement...")
            plot_elastic_dt_history(elastic_obj, dt)
            plot_watermark_stats(wmark_obj)
            print_enhancement_report(wmark_obj, elastic_obj, BUFFER_SIZE)

    print(f"\n[SELESAI] Semua grafik tersimpan di folder output/")
