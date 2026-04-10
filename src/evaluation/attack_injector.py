"""
src/evaluation/attack_injector.py
==================================
Menyuntikkan log serangan buatan ke dataset raw Wazuh untuk mendapatkan
ground truth yang valid bagi evaluasi PR-AUC, F0.5, dan FNR.

PRINSIP: injeksi dilakukan di dataset RAW (sebelum RBTA), bukan di meta-alert.
Alasannya: pipeline RBTA + feature_engineering harus memproses data mentah
agar fitur f10-f13 yang dihasilkan valid dan tidak manipulatif secara akademis.

Kolom tambahan yang ditambahkan ke setiap baris:
  is_synthetic (int 0/1) — 1 jika baris ini hasil injeksi
  scenario_id  (str)     — "A", "B", "C", atau "" untuk baris asli

Setelah melewati RBTA, meta-alert akan mendapat kolom:
  max_is_synthetic — max() dari is_synthetic dalam bucket (1 jika ada alert injeksi)
  scenario_id      — scenario_id dari alert injeksi dalam bucket (jika ada)

Tiga skenario:
  A  SSH Brute-Force        — volume tinggi, rule_group tunggal, satu agent
                              → menguji f1 (alert_count), f2 (max_severity)
                              → entropy rendah (f10): hanya satu jenis log

  B  Multi-Stage APT        — beberapa rule_group berurutan dalam kill chain
                              → menguji f10 (rule_group_entropy), f11 (tactic_progression)
                              → urutan: authentication_failed → syscheck_file → attack

  C  Cross-Agent Lateral    — satu IP eksternal menyerang 3 agent berbeda
     Movement                 dalam 1 jam
                              → menguji f13 (cross_agent_spread)

Cara penggunaan:
    python -m src.evaluation.attack_injector \\
        --input  data/raw/rbta_ready_ALL.csv \\
        --output data/injected/rbta_injected.csv \\
        --scenarios A B C

    # Hanya skenario B dan C:
    python -m src.evaluation.attack_injector \\
        --input  data/raw/rbta_ready_ALL.csv \\
        --output data/injected/rbta_injected.csv \\
        --scenarios B C
"""

import argparse
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RANDOM_SEED = 42

# ── Konstanta domain INSTIKI ──────────────────────────────────────────────────

KNOWN_AGENTS: dict[str, str] = {
    "soc-1":         "001",
    "pusatkarir":    "002",
    "dfir-iris":     "003",
    "siput":         "004",
    "proxy-manager": "005",
    "e-kuesioner":   "006",
    "sads":          "007",
    "dvwa":          "008",
}

# IP eksternal buatan — range publik yang tidak ada di dataset asli
SYNTHETIC_ATTACKER_IPS: list[str] = [
    "185.220.101.45",
    "45.142.212.100",
    "194.165.16.11",
]

# Template rule per skenario — berdasarkan rule_id yang umum di Wazuh
SCENARIO_A_TEMPLATE = {
    "rule_group_primary": "authentication_failed",
    "rule_level":         10,
    "rule_id":            "5716",
    "srcip_type":         "external",
    "criticality_score":  1,
    "has_mitre":          0,
    "has_critical_mitre": 0,
    "mitre_tactic":       "",
}

# Skenario B: 3 fase berurutan dalam kill chain
SCENARIO_B_PHASES: list[dict] = [
    {
        "rule_group_primary": "authentication_failed",
        "rule_level":         8,
        "rule_id":            "5720",
        "has_mitre":          1,
        "has_critical_mitre": 0,
        "mitre_tactic":       "Discovery",
        "n_alerts":           15,
        "interval_seconds":   4,
    },
    {
        "rule_group_primary": "web",
        "rule_level":         11,
        "rule_id":            "31101",
        "has_mitre":          1,
        "has_critical_mitre": 1,
        "mitre_tactic":       "Initial Access",
        "n_alerts":           8,
        "interval_seconds":   6,
    },
    {
        "rule_group_primary": "syscheck_file",
        "rule_level":         7,
        "rule_id":            "553",
        "has_mitre":          1,
        "has_critical_mitre": 1,
        "mitre_tactic":       "Persistence",
        "n_alerts":           5,
        "interval_seconds":   8,
    },
]

# Skenario C: IP yang sama menyerang 3 agent berbeda
SCENARIO_C_TARGETS: list[str] = ["siput", "proxy-manager", "sads"]
SCENARIO_C_RULE = {
    "rule_group_primary": "authentication_failed",
    "rule_level":         10,
    "rule_id":            "5763",
    "has_mitre":          1,
    "has_critical_mitre": 0,
    "mitre_tactic":       "Lateral Movement",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: buat satu baris synthetic
# ══════════════════════════════════════════════════════════════════════════════

def _make_row(
    template:    dict,
    ts:          datetime,
    agent_name:  str,
    srcip:       str,
    scenario_id: str,
) -> dict:
    """Bangun satu baris raw alert sintetis berformat rbta_ready_ALL.csv."""
    agent_id = KNOWN_AGENTS.get(agent_name, "999")
    return {
        "wazuh_alert_id":      f"SYN-{scenario_id}-{random.randint(10**9, 10**10)}",
        "timestamp_utc":       ts.strftime("%Y-%m-%d %H:%M:%S.%f+00:00"),
        "agent_id":            agent_id,
        "agent_name":          agent_name,
        "rule_group_primary":  template["rule_group_primary"],
        "rule_level":          template["rule_level"],
        "rule_id":             template["rule_id"],
        "srcip":               srcip,
        "srcip_type":          template.get("srcip_type", "external"),
        "criticality_score":   template.get("criticality_score", 2),
        "has_mitre":           template.get("has_mitre", 0),
        "has_critical_mitre":  template.get("has_critical_mitre", 0),
        "mitre_tactic":        template.get("mitre_tactic", ""),
        # Kolom ground truth
        "is_synthetic":        1,
        "scenario_id":         scenario_id,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Skenario A: SSH Brute-Force
# ══════════════════════════════════════════════════════════════════════════════

def inject_scenario_a(
    df:         pd.DataFrame,
    n_alerts:   int = 60,
    interval_s: int = 2,
) -> list[dict]:
    """
    Injeksi SSH Brute-Force ke satu agent selama ~2 menit.

    Desain:
      - Rule group tunggal: authentication_failed
      - Volume tinggi: n_alerts dalam interval_s detik tiap alert
      - Severity 10 (high) — memicu Q1 Decision Matrix
      - Entropy rendah (f10): semua log dari grup yang sama
      - Tidak ada sinyal MITRE — menguji apakah FP Gate berjalan
        (volume tinggi + severity tinggi → tetap ESCALATE)

    Parameter
    ----------
    df       : DataFrame asli (untuk mengambil rentang waktu yang realistis)
    n_alerts : jumlah alert yang diinjeksi
    interval_s: interval antar alert dalam detik
    """
    random.seed(RANDOM_SEED)

    # Ambil timestamp tengah dataset supaya tidak mepet di tepi
    ts_min = pd.to_datetime(df["timestamp_utc"]).min()
    ts_max = pd.to_datetime(df["timestamp_utc"]).max()
    mid_ts = ts_min + (ts_max - ts_min) * 0.3
    base_ts = mid_ts.to_pydatetime().replace(microsecond=0)

    attacker_ip = SYNTHETIC_ATTACKER_IPS[0]
    target_agent = "pusatkarir"

    rows = []
    for i in range(n_alerts):
        ts  = base_ts + timedelta(seconds=i * interval_s)
        row = _make_row(SCENARIO_A_TEMPLATE, ts, target_agent, attacker_ip, "A")
        rows.append(row)

    log.info(
        "[SCENARIO A] SSH Brute-Force: %d alert → agent=%s  IP=%s  start=%s",
        n_alerts, target_agent, attacker_ip, base_ts.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Skenario B: Multi-Stage APT
# ══════════════════════════════════════════════════════════════════════════════

def inject_scenario_b(
    df:            pd.DataFrame,
    phase_gap_min: int = 5,
) -> list[dict]:
    """
    Injeksi serangan multi-stage mengikuti kill chain MITRE ATT&CK.

    Urutan fase:
      Fase 1 (Discovery)     — authentication_failed, level 8,  15 alert
      Fase 2 (Initial Access)— web/SQL injection,    level 11,  8 alert
      Fase 3 (Persistence)   — syscheck_file,        level 7,   5 alert

    Desain:
      - Setiap fase dipisahkan phase_gap_min menit
      - tactic_progression_score (f11) harus mendekati 1.0
      - rule_group_entropy (f10) tinggi karena 3 grup berbeda
      - Ini adalah skenario yang paling representatif untuk APT nyata
    """
    random.seed(RANDOM_SEED + 1)

    ts_min = pd.to_datetime(df["timestamp_utc"]).min()
    ts_max = pd.to_datetime(df["timestamp_utc"]).max()
    base_ts = (ts_min + (ts_max - ts_min) * 0.5).to_pydatetime().replace(microsecond=0)

    attacker_ip  = SYNTHETIC_ATTACKER_IPS[1]
    target_agent = "sads"

    rows = []
    phase_start = base_ts

    for phase in SCENARIO_B_PHASES:
        n        = phase["n_alerts"]
        interval = phase["interval_seconds"]
        for i in range(n):
            ts  = phase_start + timedelta(seconds=i * interval)
            row = _make_row(phase, ts, target_agent, attacker_ip, "B")
            rows.append(row)
        phase_start += timedelta(minutes=phase_gap_min)

    n_total = len(rows)
    log.info(
        "[SCENARIO B] Multi-Stage APT: %d alert  agent=%s  IP=%s  "
        "fases=[Discovery, Initial Access, Persistence]",
        n_total, target_agent, attacker_ip,
    )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Skenario C: Cross-Agent Lateral Movement
# ══════════════════════════════════════════════════════════════════════════════

def inject_scenario_c(
    df:              pd.DataFrame,
    n_per_agent:     int = 20,
    inter_agent_min: int = 15,
) -> list[dict]:
    """
    Injeksi lateral movement: satu IP menyerang 3 agent berbeda berurutan.

    Desain:
      - IP yang sama (SYNTHETIC_ATTACKER_IPS[2]) muncul di siput →
        proxy-manager → sads dengan jarak inter_agent_min menit
      - cross_agent_spread (f13) harus = 3 untuk meta-alert dari IP ini
      - Severity sedang (level 10), ada sinyal MITRE Lateral Movement
    """
    random.seed(RANDOM_SEED + 2)

    ts_min = pd.to_datetime(df["timestamp_utc"]).min()
    ts_max = pd.to_datetime(df["timestamp_utc"]).max()
    base_ts = (ts_min + (ts_max - ts_min) * 0.7).to_pydatetime().replace(microsecond=0)

    attacker_ip = SYNTHETIC_ATTACKER_IPS[2]
    rows        = []

    for agent_idx, agent_name in enumerate(SCENARIO_C_TARGETS):
        agent_start = base_ts + timedelta(minutes=agent_idx * inter_agent_min)
        for i in range(n_per_agent):
            ts  = agent_start + timedelta(seconds=i * 3)
            row = _make_row(SCENARIO_C_RULE, ts, agent_name, attacker_ip, "C")
            rows.append(row)

    n_total = len(rows)
    log.info(
        "[SCENARIO C] Lateral Movement: %d alert  IP=%s  targets=%s",
        n_total, attacker_ip, SCENARIO_C_TARGETS,
    )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline injeksi
# ══════════════════════════════════════════════════════════════════════════════

def run_injection(
    input_path:  str | Path,
    output_path: str | Path,
    scenarios:   list[str] = ("A", "B", "C"),
) -> pd.DataFrame:
    """
    Baca CSV asli, injeksi skenario yang dipilih, simpan CSV baru.

    Kolom yang ditambahkan ke SEMUA baris (asli maupun sintetis):
      is_synthetic (int) — 0 untuk asli, 1 untuk sintetis
      scenario_id  (str) — "" untuk asli, "A"/"B"/"C" untuk sintetis

    Parameter
    ----------
    input_path  : path ke rbta_ready_ALL.csv asli
    output_path : path output rbta_injected.csv
    scenarios   : daftar skenario yang diinjeksi

    Returns
    -------
    df_injected : DataFrame gabungan (asli + sintetis), disortir temporal
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    log.info("[INJECT] Membaca: %s", input_path)
    df_raw = pd.read_csv(input_path, low_memory=False)
    n_original = len(df_raw)
    log.info("[INJECT] Dataset asli: %d baris", n_original)

    # Tambahkan kolom ground truth ke baris asli
    df_raw["is_synthetic"] = 0
    df_raw["scenario_id"]  = ""

    # Kumpulkan baris sintetis
    synthetic_rows: list[dict] = []

    if "A" in scenarios:
        synthetic_rows.extend(inject_scenario_a(df_raw))
    if "B" in scenarios:
        synthetic_rows.extend(inject_scenario_b(df_raw))
    if "C" in scenarios:
        synthetic_rows.extend(inject_scenario_c(df_raw))

    if not synthetic_rows:
        log.warning("Tidak ada skenario yang diinjeksi. Output = input asli.")
        df_raw.to_csv(output_path, index=False)
        return df_raw

    df_synthetic = pd.DataFrame(synthetic_rows)

    # Pastikan semua kolom yang ada di df_raw juga ada di df_synthetic
    for col in df_raw.columns:
        if col not in df_synthetic.columns:
            df_synthetic[col] = None

    # Gabungkan dan sort temporal
    df_injected = (
        pd.concat([df_raw, df_synthetic[df_raw.columns]], ignore_index=True)
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    n_synth = df_synthetic["is_synthetic"].sum()
    pct     = n_synth / len(df_injected) * 100

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_injected.to_csv(output_path, index=False)

    log.info("[INJECT] Selesai.")
    log.info("  Baris asli       : %d", n_original)
    log.info("  Baris sintetis   : %d (%.2f%% dari total)", n_synth, pct)
    log.info("  Total            : %d", len(df_injected))
    log.info("  Output disimpan  : %s", output_path)

    _print_injection_summary(df_injected)
    return df_injected


def _print_injection_summary(df: pd.DataFrame) -> None:
    log.info("\n[INJECT SUMMARY]")
    for sid in ["A", "B", "C"]:
        subset = df[df["scenario_id"] == sid]
        if len(subset) == 0:
            continue
        ts_min = subset["timestamp_utc"].min()
        ts_max = subset["timestamp_utc"].max()
        agents = subset["agent_name"].unique().tolist()
        ips    = subset["srcip"].unique().tolist()
        log.info(
            "  Scenario %s: %d baris  agents=%s  IPs=%s  window=[%s → %s]",
            sid, len(subset), agents, ips, ts_min, ts_max,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Propagasi label ke meta-alert (dipanggil setelah RBTA)
# ══════════════════════════════════════════════════════════════════════════════

def propagate_labels(
    df_meta:         pd.DataFrame,
    df_raw_injected: pd.DataFrame,
    alert_index_map: dict[int, list[int]],
) -> pd.DataFrame:
    """
    Propagasi label is_synthetic dari raw alerts ke meta-alert via alert_index_map.

    Untuk setiap meta_id, cek apakah ada raw alert dengan is_synthetic == 1
    dalam bucket tersebut. Jika ada, meta-alert diberi label positif (1).

    Ini adalah cara yang benar untuk mendapatkan ground truth di level
    meta-alert tanpa memanipulasi data meta-alert secara langsung.

    Parameter
    ----------
    df_meta         : DataFrame meta-alert hasil run_rbta()
    df_raw_injected : DataFrame raw injected (dengan kolom is_synthetic, scenario_id)
    alert_index_map : { meta_id: [row_index_in_df_raw] }

    Returns
    -------
    df_meta dengan kolom tambahan:
      ground_truth  (int 0/1) — 1 jika ada alert sintetis dalam bucket
      scenario_id   (str)     — scenario_id dari alert sintetis dalam bucket
    """
    df = df_meta.copy()

    if "is_synthetic" not in df_raw_injected.columns:
        log.warning(
            "Kolom is_synthetic tidak ditemukan di df_raw_injected. "
            "Pastikan run_injection() sudah dijalankan sebelum RBTA."
        )
        df["ground_truth"] = 0
        df["scenario_id"]  = ""
        return df

    ground_truth: list[int] = []
    scenario_ids: list[str] = []

    for _, row in df.iterrows():
        mid      = row["meta_id"]
        idx_list = alert_index_map.get(mid, [])

        if not idx_list:
            ground_truth.append(0)
            scenario_ids.append("")
            continue

        # Ambil baris raw yang tergabung dalam bucket ini
        raw_subset = df_raw_injected.iloc[
            [i for i in idx_list if i < len(df_raw_injected)]
        ]

        has_synthetic = int(raw_subset["is_synthetic"].any())
        scenario      = ""
        if has_synthetic:
            synth_rows = raw_subset[raw_subset["is_synthetic"] == 1]
            scenario   = synth_rows["scenario_id"].iloc[0] if len(synth_rows) > 0 else ""

        ground_truth.append(has_synthetic)
        scenario_ids.append(scenario)

    df["ground_truth"] = ground_truth
    df["scenario_id"]  = scenario_ids

    n_pos = sum(ground_truth)
    log.info(
        "[PROPAGATE] Ground truth selesai: %d meta-alert positif (%.1f%%) dari %d total",
        n_pos, n_pos / len(df) * 100, len(df),
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Synthetic attack injection untuk ground truth evaluasi RBTA."
    )
    parser.add_argument(
        "--input",  default="data/raw/rbta_ready_ALL.csv",
        help="Path ke dataset raw (rbta_ready_ALL.csv)",
    )
    parser.add_argument(
        "--output", default="data/injected/rbta_injected.csv",
        help="Path output dataset yang sudah diinjeksi",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Daftar skenario yang diinjeksi (default: A B C)",
    )
    args = parser.parse_args()

    run_injection(
        input_path  = args.input,
        output_path = args.output,
        scenarios   = args.scenarios,
    )