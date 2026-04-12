"""
robustness.py  —  Noise Robustness Test (Landauer et al. 2022, Section 6.9)
============================================================================
Mengimplementasikan evaluasi robustness dari Landauer et al. (2022):

  Landauer: "We evaluate robustness by randomly duplicating alerts and
  uniformly distributing them over input data. Adjusting total alerts
  added allows setting noise intensity."

  Temuan Landauer: TPR dan F1 menurun drastis saat noise ~10 alert/menit
  karena noise alert menghubungkan dua grup yang tidak berkaitan.

  Hipotesis RBTA: Karena RBTA menggunakan konteks (agent_id + rule_group),
  tidak hanya waktu, RBTA seharusnya LEBIH ROBUST dari Landauer's pure
  time-window approach. Ini adalah klaim yang dapat dibuktikan secara empiris.

Alur evaluasi (mirror Figure 16 Landauer):
  1. Baseline: jalankan RBTA tanpa noise → ARR_0, n_meta_0
  2. Untuk setiap noise_rate ∈ {0.05, 0.10, 0.20, 0.30}:
     a. Suntikkan noise alerts acak ke df_raw
     b. Jalankan RBTA pada data ber-noise
     c. Catat ARR, n_meta, n_noise_absorbed (berapa noise masuk bucket valid)
  3. Plot: noise_rate vs ARR degradation (Figure 16 analog)
  4. Plot: noise_rate vs noise absorption rate

Metrik kunci:
  ARR_degradation   : ARR_0 - ARR_noise  (seberapa turun ARR akibat noise)
  noise_absorption  : noise yang masuk bucket valid / total noise
                      Tinggi = buruk (noise merusak meta-alert valid)
                      Rendah = baik (RBTA mengisolasi noise ke bucket sendiri)
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Konfigurasi noise test ────────────────────────────────────────────────────
NOISE_RATES        = [0.0, 0.05, 0.10, 0.20, 0.30]   # fraksi dari total alert
DELTA_T_FIXED      = 15                                 # Δt optimal dari sensitivity
BUFFER_SIZE_FIXED  = 50
MAX_LATENESS_FIXED = 60.0
RANDOM_SEED        = 42

# Palette warna konsisten dengan modul lain
C_PRIMARY = "#534AB7"
C_ACCENT  = "#1D9E75"
C_WARN    = "#BA7517"
C_RED     = "#C94040"
C_GRAY    = "#888780"


# ══════════════════════════════════════════════════════════════════════════════
# Noise Injection (inti dari robustness test)
# ══════════════════════════════════════════════════════════════════════════════

def inject_noise(
    df:         pd.DataFrame,
    noise_rate: float = 0.10,
    seed:       int   = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Suntikkan alert noise acak ke dataset raw.

    Desain mengikuti Landauer Section 6.9:
    "We randomly duplicate alerts and uniformly distribute them
    over the input data."

    Perbedaan dengan Landauer: noise kita gunakan rule_group ACAK
    (tidak hanya duplikasi) untuk mensimulasikan false positive IDS
    yang lebih realistis di lingkungan Wazuh.

    Alert noise memiliki:
    - Timestamp: tersebar merata di rentang dataset
    - agent_name: dipilih acak dari agen yang ada
    - rule_group_primary: dipilih acak dari rule_group yang ada
    - rule_level: 1-4 (severity rendah, ciri khas false positive)
    - Semua field MITRE: kosong (noise bukan serangan nyata)

    Parameters
    ----------
    df         : DataFrame raw alerts (output preprocessing_01)
    noise_rate : Fraksi noise = n_noise / n_original. 0.10 = 10% noise.
    seed       : Random seed untuk reprodusibilitas.

    Returns
    -------
    pd.DataFrame gabungan asli + noise, diurutkan temporal.
                 Kolom tambahan: is_noise (0/1)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Tandai data asli
    df = df.copy()
    df["is_noise"] = 0

    n_original = len(df)
    n_noise    = int(n_original * noise_rate)

    if n_noise == 0:
        log.info("[NOISE] noise_rate=%.2f → 0 noise alerts disuntikkan", noise_rate)
        return df

    # Ambil pool nilai dari dataset asli
    rule_groups = df["rule_groups"].dropna().unique().tolist()
    agents      = df["agent_name"].dropna().unique().tolist()
    agent_ids   = df["agent_id"].dropna().unique().tolist()

    # Rentang timestamp
    ts_min = pd.to_datetime(df["timestamp"]).min()
    ts_max = pd.to_datetime(df["timestamp"]).max()
    ts_range_sec = (ts_max - ts_min).total_seconds()

    noise_rows = []
    for _ in range(n_noise):
        # Timestamp tersebar merata
        offset_sec = random.uniform(0, ts_range_sec)
        ts_noise   = ts_min + pd.Timedelta(seconds=offset_sec)

        # Pilih agent dan rule_group acak
        agent_name = random.choice(agents)
        agent_id   = random.choice(agent_ids)
        rule_group = random.choice(rule_groups)

        noise_rows.append({
            "timestamp":           ts_noise,
            "agent_id":            agent_id,
            "agent_name":          agent_name,
            "rule_groups":         rule_group,
            "rule_level":          random.randint(1, 4),  # severity rendah
            "rule_id":             random.randint(1000, 9999),
            "srcip":               None,
            "srcip_type":          "none",
            "criticality_score":   1,
            "has_mitre":           0,
            "has_critical_mitre":  0,
            "rule_firedtimes":     1,
            "mitre_tactic":        "",
            "is_noise":            1,
        })

    df_noise = pd.DataFrame(noise_rows)

    # Pastikan kolom yang ada di df_noise tersedia di df
    for col in df.columns:
        if col not in df_noise.columns:
            df_noise[col] = None

    df_out = (
        pd.concat([df[list(df_noise.columns)], df_noise], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    log.info(
        "[NOISE] Injeksi %.0f%% noise → %d alert noise disuntikkan "
        "(total dataset: %d → %d)",
        noise_rate * 100, n_noise, n_original, len(df_out),
    )
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Noise Robustness Test
# ══════════════════════════════════════════════════════════════════════════════

def noise_robustness_test(
    df_raw:           pd.DataFrame,
    delta_t_minutes:  int   = DELTA_T_FIXED,
    buffer_size:      int   = BUFFER_SIZE_FIXED,
    max_lateness_sec: float = MAX_LATENESS_FIXED,
    noise_rates:      list  = NOISE_RATES,
) -> pd.DataFrame:
    """
    Jalankan RBTA pada dataset dengan berbagai tingkat noise,
    ukur degradasi ARR dan noise absorption rate.

    Ini adalah implementasi Landauer Section 6.9 untuk konteks RBTA.

    Returns
    -------
    pd.DataFrame dengan kolom:
      noise_rate         : fraksi noise yang disuntikkan
      n_original         : jumlah alert asli
      n_noise            : jumlah alert noise
      n_total            : n_original + n_noise
      n_meta             : jumlah meta-alert yang dihasilkan
      arr_pct            : Alert Reduction Rate (%)
      arr_degradation    : ARR_baseline - ARR_noise (penurunan ARR)
      noise_absorbed     : berapa noise alert masuk ke bucket valid (bukan isolat)
      noise_absorption_rt: noise_absorbed / n_noise (%)
      exec_time_ms       : waktu eksekusi RBTA (ms)
    """
    from src.engine.rbta_core import run_rbta

    log.info("[ROBUSTNESS] Memulai noise robustness test ...")
    log.info("  Δt=%d menit | buffer=%d | noise_rates=%s",
             delta_t_minutes, buffer_size, noise_rates)

    results = []
    arr_baseline = None

    for rate in noise_rates:
        # Suntikkan noise
        if rate > 0:
            df_noisy = inject_noise(df_raw, noise_rate=rate)
        else:
            df_noisy = df_raw.copy()
            df_noisy["is_noise"] = 0

        n_original = int((df_noisy["is_noise"] == 0).sum())
        n_noise    = int((df_noisy["is_noise"] == 1).sum())
        n_total    = len(df_noisy)

        # Jalankan RBTA
        t0 = time.perf_counter()
        df_meta, _, idx_map, _, _ = run_rbta(
            df_noisy,
            delta_t_minutes  = delta_t_minutes,
            buffer_size      = buffer_size,
            max_lateness_sec = max_lateness_sec,
            enable_adaptive  = False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        n_meta = len(df_meta)
        arr    = (1 - n_meta / n_original) * 100 if n_original > 0 else 0.0

        if rate == 0.0:
            arr_baseline = arr

        arr_degradation = (arr_baseline - arr) if arr_baseline is not None else 0.0

        # Hitung noise absorption: berapa noise masuk ke bucket yg mengandung alert asli
        noise_absorbed = 0
        if n_noise > 0 and "is_noise" in df_noisy.columns:
            noise_indices = df_noisy[df_noisy["is_noise"] == 1].index.tolist()
            noise_idx_set = set(noise_indices)
            for mid, idx_list in idx_map.items():
                idx_set = set(idx_list)
                has_real  = bool(idx_set - noise_idx_set)
                has_noise = bool(idx_set & noise_idx_set)
                if has_real and has_noise:
                    # Bucket valid yang terkontaminasi noise
                    noise_absorbed += len(idx_set & noise_idx_set)

        noise_absorption_rt = (noise_absorbed / n_noise * 100) if n_noise > 0 else 0.0

        results.append({
            "noise_rate":          rate,
            "n_original":          n_original,
            "n_noise":             n_noise,
            "n_total":             n_total,
            "n_meta":              n_meta,
            "arr_pct":             round(arr, 2),
            "arr_degradation":     round(arr_degradation, 2),
            "noise_absorbed":      noise_absorbed,
            "noise_absorption_rt": round(noise_absorption_rt, 2),
            "exec_time_ms":        round(elapsed_ms, 2),
        })

        log.info(
            "  noise=%.0f%%  n_noise=%d  n_meta=%d  ARR=%.2f%%  "
            "degradation=%.2f%%  absorption=%.2f%%  t=%.1fms",
            rate * 100, n_noise, n_meta, arr,
            arr_degradation, noise_absorption_rt, elapsed_ms,
        )

    df_result = pd.DataFrame(results)
    log.info("[ROBUSTNESS] Selesai. Ringkasan:\n%s",
             df_result[["noise_rate", "arr_pct", "arr_degradation",
                        "noise_absorption_rt"]].to_string(index=False))
    return df_result


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_robustness(
    df_robust:  pd.DataFrame,
    output_dir: str = "reports/figures",
) -> None:
    """
    Plot Figure 16 analog dari Landauer et al. (2022):
    Pengaruh noise terhadap ARR dan noise absorption rate.

    Panel kiri : noise_rate vs ARR (+ degradation)
    Panel kanan: noise_rate vs noise absorption rate
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Robustness Analysis: Pengaruh Noise Alerts terhadap RBTA\n"
        "(Reproduksi Figure 16 — Landauer et al., 2022)",
        fontsize=11, y=1.01,
    )

    # Panel kiri: ARR vs noise rate
    x = df_robust["noise_rate"] * 100

    ax1.plot(x, df_robust["arr_pct"], color=C_PRIMARY, marker="o",
             linewidth=2.2, markersize=7, label="ARR (%)", zorder=3)
    ax1.fill_between(x, df_robust["arr_pct"], alpha=0.08, color=C_PRIMARY)

    ax1b = ax1.twinx()
    ax1b.plot(x, df_robust["arr_degradation"], color=C_RED, marker="s",
              linewidth=1.5, markersize=5, linestyle="--",
              label="Degradasi ARR (pp)", zorder=2)
    ax1b.set_ylabel("Degradasi ARR (percentage points)", fontsize=10, color=C_RED)
    ax1b.tick_params(axis="y", labelcolor=C_RED)

    ax1.set_xlabel("Noise Rate (% dari total alert)", fontsize=10)
    ax1.set_ylabel("Alert Reduction Rate (%)", fontsize=10, color=C_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    for _, row in df_robust.iterrows():
        ax1.annotate(f"{row['arr_pct']:.1f}%",
                     xy=(row["noise_rate"] * 100, row["arr_pct"]),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=8, color=C_PRIMARY)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title("ARR vs Tingkat Noise", fontsize=10, pad=8)

    # Panel kanan: noise absorption rate
    ax2.bar(x, df_robust["noise_absorption_rt"], color=C_WARN, alpha=0.80,
            width=3.5, zorder=3, label="Noise absorption (%)")
    ax2.set_xlabel("Noise Rate (% dari total alert)", fontsize=10)
    ax2.set_ylabel("Noise Absorption Rate (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_title("Kontaminasi Bucket Valid oleh Noise", fontsize=10, pad=8)

    for i, row in df_robust.iterrows():
        if row["n_noise"] > 0:
            ax2.text(row["noise_rate"] * 100, row["noise_absorption_rt"] + 0.5,
                     f"{row['noise_absorption_rt']:.1f}%",
                     ha="center", fontsize=8, color=C_WARN)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "robustness_noise_test.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("[PLOT] Robustness plot tersimpan: %s", out_path)
    plt.close(fig)


def print_robustness_table(df_robust: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("  TABEL ROBUSTNESS — Pengaruh Noise terhadap RBTA")
    print("  (Reproduksi Landauer et al. 2022, Section 6.9)")
    print("=" * 80)
    print(f"  {'Noise Rate':<12} {'N Noise':<10} {'N Meta':<10} "
          f"{'ARR (%)':<10} {'Degradasi':<12} {'Absorption':<12} {'Exec (ms)'}")
    print("  " + "-" * 76)
    for _, row in df_robust.iterrows():
        print(
            f"  {row['noise_rate']*100:>8.0f}%   {int(row['n_noise']):<10} "
            f"{int(row['n_meta']):<10} {row['arr_pct']:<10.2f} "
            f"{row['arr_degradation']:<12.2f} {row['noise_absorption_rt']:<12.2f} "
            f"{row['exec_time_ms']:.1f}"
        )
    print("=" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/rbta_ready_ALL.csv"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "reports/figures"

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.etl.preprocessing_01 import load_and_prepare

    log.info("Loading: %s", csv_path)
    df_raw = load_and_prepare(csv_path)

    df_robust = noise_robustness_test(df_raw)
    print_robustness_table(df_robust)
    plot_robustness(df_robust, output_dir=out_dir)

    df_robust.to_csv("reports/data_quality/robustness_results.csv", index=False)
    log.info("Hasil disimpan: reports/data_quality/robustness_results.csv")