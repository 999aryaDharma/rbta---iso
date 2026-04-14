"""
attack_injector.py  —  Synthetic Attack Injection + Noise Injection
====================================================================
Dua fungsi utama:

  [1] Synthetic Attack Injection — untuk ground truth evaluasi (PR-AUC, FNR)
      Tiga skenario serangan sintetis:
        A: SSH Brute-Force (volume tinggi, satu agent)
        B: Multi-Stage APT (kill chain: Discovery → Initial Access → Persistence)
        C: Cross-Agent Lateral Movement (satu IP, 3 agent)

  [2] Noise Injection — untuk robustness test (Landauer Section 6.9)
      Alert acak tersebar merata, severity rendah, tanpa MITRE signal.
      Mengukur ketahanan RBTA terhadap IDS false-positive flooding.

Referensi: Landauer et al. (2022), Sections 6.6 dan 6.9.
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

# ── Domain constants ──────────────────────────────────────────────────────────
KNOWN_AGENTS: dict[str, str] = {
    "soc-1":         "001",
    "pusatkarir":    "002",
    "dfir-iris":     "003",
    "siput":         "004",
    "proxy-manager": "005",
    "e-kuesioner":   "006",
    "sads":          "007",
    "DVWA":          "008",
}

SYNTHETIC_ATTACKER_IPS: list[str] = [
    "185.220.101.45",
    "45.142.212.100",
    "194.165.16.11",
]

SCENARIO_A_TEMPLATE = {
    "rule_group_primary": "authentication_failed",
    "rule_level": 10,
    "rule_id":    "5716",
    "srcip_type": "external",
    "criticality_score": 1,
    "has_mitre": 0, "has_critical_mitre": 0,
    "mitre_tactic": "",
}

SCENARIO_B_PHASES: list[dict] = [
    {"rule_group_primary": "authentication_failed", "rule_level": 8, "rule_id": "5720",
     "has_mitre": 1, "has_critical_mitre": 0, "mitre_tactic": "Discovery",
     "n_alerts": 2000, "interval_seconds": 4},
    {"rule_group_primary": "web",  "rule_level": 11, "rule_id": "31101",
     "has_mitre": 1, "has_critical_mitre": 1, "mitre_tactic": "Initial Access",
     "n_alerts": 1500, "interval_seconds": 6},
    {"rule_group_primary": "syscheck_file", "rule_level": 7, "rule_id": "553",
     "has_mitre": 1, "has_critical_mitre": 1, "mitre_tactic": "Persistence",
     "n_alerts": 1000, "interval_seconds": 8},
]

SCENARIO_C_TARGETS: list[str] = [
    "siput", "proxy-manager", "sads", "e-kuesioner", "dvwa",
    "dfir-iris", "pusatkarir", "soc-1"
]
SCENARIO_C_RULE = {
    "rule_group_primary": "authentication_failed",
    "rule_level": 10, "rule_id": "5763",
    "has_mitre": 1, "has_critical_mitre": 0, "mitre_tactic": "Lateral Movement",
}


# ══════════════════════════════════════════════════════════════════════════════
# [1] SYNTHETIC ATTACK INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def _make_row(template: dict, ts: datetime, agent_name: str,
              srcip: str, scenario_id: str) -> dict:
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
        "is_synthetic":        1,
        "scenario_id":         scenario_id,
    }


def inject_scenario_a(df: pd.DataFrame, n_alerts: int = 50000,
                      interval_s: int = 2) -> list[dict]:
    random.seed(RANDOM_SEED)
    ts_min = pd.to_datetime(df["timestamp_utc"], format="ISO8601", utc=True).min()
    ts_max = pd.to_datetime(df["timestamp_utc"], format="ISO8601", utc=True).max()
    mid_ts = ts_min + (ts_max - ts_min) * 0.3
    base_ts = mid_ts.to_pydatetime().replace(tzinfo=None, microsecond=0)
    rows = []
    for i in range(n_alerts):
        ts  = base_ts + timedelta(seconds=i * interval_s)
        row = _make_row(SCENARIO_A_TEMPLATE, ts, "pusatkarir",
                        SYNTHETIC_ATTACKER_IPS[0], "A")
        rows.append(row)
    log.info("[SCENARIO A] SSH Brute-Force: %d alert → pusatkarir", n_alerts)
    return rows


def inject_scenario_b(df: pd.DataFrame, phase_gap_min: int = 5) -> list[dict]:
    random.seed(RANDOM_SEED + 1)
    ts_min = pd.to_datetime(df["timestamp_utc"], format="ISO8601", utc=True).min()
    ts_max = pd.to_datetime(df["timestamp_utc"], format="ISO8601", utc=True).max()
    base_ts = (ts_min + (ts_max - ts_min) * 0.5).to_pydatetime().replace(
               tzinfo=None, microsecond=0)
    rows, phase_start = [], base_ts
    for phase in SCENARIO_B_PHASES:
        for i in range(phase["n_alerts"]):
            ts  = phase_start + timedelta(seconds=i * phase["interval_seconds"])
            row = _make_row(phase, ts, "sads", SYNTHETIC_ATTACKER_IPS[1], "B")
            rows.append(row)
        phase_start += timedelta(minutes=phase_gap_min)
    log.info("[SCENARIO B] Multi-Stage APT: %d alert → sads", len(rows))
    return rows


def inject_scenario_c(df: pd.DataFrame, n_per_agent: int = 50000,
                      inter_agent_min: int = 15) -> list[dict]:
    random.seed(RANDOM_SEED + 2)
    ts_min  = pd.to_datetime(df["timestamp_utc"], format="ISO8601", utc=True).min()
    ts_max  = pd.to_datetime(df["timestamp_utc"], format="ISO8601", utc=True).max()
    base_ts = (ts_min + (ts_max - ts_min) * 0.7).to_pydatetime().replace(
               tzinfo=None, microsecond=0)
    rows = []
    for idx, agent_name in enumerate(SCENARIO_C_TARGETS):
        agent_start = base_ts + timedelta(minutes=idx * inter_agent_min)
        for i in range(n_per_agent):
            ts  = agent_start + timedelta(seconds=i * 3)
            row = _make_row(SCENARIO_C_RULE, ts, agent_name,
                            SYNTHETIC_ATTACKER_IPS[2], "C")
            rows.append(row)
    log.info("[SCENARIO C] Lateral Movement: %d alert → %s", len(rows), SCENARIO_C_TARGETS)
    return rows


def run_injection(input_path: str | Path, output_path: str | Path,
                  scenarios: list[str] = ("A", "B", "C")) -> pd.DataFrame:
    """
    Baca CSV asli, injeksi skenario, simpan CSV baru dengan kolom ground truth.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    log.info("[INJECT] Membaca: %s", input_path)
    df_raw             = pd.read_csv(input_path, low_memory=False)
    df_raw["is_synthetic"] = 0
    df_raw["scenario_id"]  = ""

    synthetic_rows: list[dict] = []
    if "A" in scenarios:
        synthetic_rows.extend(inject_scenario_a(df_raw))
    if "B" in scenarios:
        synthetic_rows.extend(inject_scenario_b(df_raw))
    if "C" in scenarios:
        synthetic_rows.extend(inject_scenario_c(df_raw))

    if not synthetic_rows:
        df_raw.to_csv(output_path, index=False)
        return df_raw

    df_synthetic = pd.DataFrame(synthetic_rows)
    for col in df_raw.columns:
        if col not in df_synthetic.columns:
            df_synthetic[col] = None

    df_injected = (
        pd.concat([df_raw, df_synthetic[df_raw.columns]], ignore_index=True)
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_injected.to_csv(output_path, index=False)
    n_synth = df_synthetic["is_synthetic"].sum()
    log.info("[INJECT] Selesai: %d asli + %d sintetis = %d total",
             len(df_raw), n_synth, len(df_injected))
    return df_injected


# ══════════════════════════════════════════════════════════════════════════════
# [2] NOISE INJECTION (Landauer Section 6.9)
# ══════════════════════════════════════════════════════════════════════════════

def inject_noise(
    df:         pd.DataFrame,
    noise_rate: float = 0.10,
    seed:       int   = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Suntikkan alert noise acak (false positive simulasi).

    Landauer (2022), Section 6.9:
    "We randomly duplicate alerts and uniformly distribute them
    over the input data to evaluate robustness."

    Implementasi RBTA: menggunakan rule_group ACAK (bukan duplikasi persis)
    untuk mensimulasikan false positive IDS yang lebih realistis.

    Parameters
    ----------
    df         : DataFrame output preprocessing_01 (sudah renamed columns)
    noise_rate : Fraksi noise = n_noise / len(df)
    seed       : Untuk reprodusibilitas

    Returns
    -------
    pd.DataFrame dengan kolom tambahan is_noise (0/1), diurutkan temporal.
    """
    random.seed(seed)
    np.random.seed(seed)

    df = df.copy()
    df["is_noise"] = 0

    n_noise = int(len(df) * noise_rate)
    if n_noise == 0:
        return df

    # Ambil pool dari data asli
    rule_groups = df["rule_groups"].dropna().unique().tolist()
    agents      = df["agent_name"].dropna().unique().tolist()
    agent_ids   = df["agent_id"].dropna().unique().tolist()

    ts_min      = df["timestamp"].min()
    ts_max      = df["timestamp"].max()
    ts_range_s  = (ts_max - ts_min).total_seconds()

    noise_rows = []
    for _ in range(n_noise):
        offset_s = random.uniform(0, ts_range_s)
        ts_noise = ts_min + pd.Timedelta(seconds=offset_s)
        noise_rows.append({
            "timestamp":           ts_noise,
            "agent_id":            random.choice(agent_ids),
            "agent_name":          random.choice(agents),
            "rule_groups":         random.choice(rule_groups),
            "rule_level":          random.randint(1, 4),
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
    # Isi kolom yang kurang
    for col in df.columns:
        if col not in df_noise.columns:
            df_noise[col] = None

    df_out = (
        pd.concat([df, df_noise[df.columns]], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    log.info("[NOISE] %.0f%% noise → %d noise alerts ditambahkan",
             noise_rate * 100, n_noise)
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# LABEL PROPAGATION
# ══════════════════════════════════════════════════════════════════════════════

def propagate_labels(df_meta: pd.DataFrame, df_raw_injected: pd.DataFrame,
                     alert_index_map: dict[int, list[int]]) -> pd.DataFrame:
    """
    Propagasi label is_synthetic dari raw alerts ke meta-alert.
    Meta-alert mendapat ground_truth=1 jika ada alert sintetis dalam bucket-nya.
    
    [FIX-A] Menggunakan wazuh_alert_ids set intersection (bukan positional index).
    Ini menghindari kontaminasi label akibat row index yang bergeser setelah preprocessing.
    """
    df = df_meta.copy()
    if "is_synthetic" not in df_raw_injected.columns:
        log.warning("[PROPAGATE] Kolom is_synthetic tidak ada — skip.")
        df["ground_truth"] = 0
        df["scenario_id"]  = ""
        return df

    # [FIX-A] Build index: wazuh_alert_id -> row di df_raw_injected
    if "wazuh_alert_id" not in df_raw_injected.columns:
        log.error("[PROPAGATE] Kolom wazuh_alert_id tidak ada — tidak bisa propagate dengan aman.")
        df["ground_truth"] = 0
        df["scenario_id"]  = ""
        return df

    # Map wazuh_alert_id -> (is_synthetic, scenario_id)
    synthetic_ids = {}
    for _, row in df_raw_injected.iterrows():
        wid = str(row.get("wazuh_alert_id", ""))
        if wid:
            synthetic_ids[wid] = {
                "is_synthetic": int(row.get("is_synthetic", 0)),
                "scenario_id": str(row.get("scenario_id", "") or ""),
            }

    log.info("[PROPAGATE] Indexed %d wazuh_alert_id dari raw data", len(synthetic_ids))

    ground_truth, scenario_ids = [], []
    for _, row in df.iterrows():
        mid = row["meta_id"]
        
        # [FIX-A] Coba ambil wazuh_alert_ids dari meta-alert
        wid_str = str(row.get("wazuh_alert_ids", "") or "")
        if not wid_str:
            # Fallback ke positional index jika wazuh_alert_ids kosong
            idx_list = alert_index_map.get(mid, [])
            if not idx_list:
                ground_truth.append(0)
                scenario_ids.append("")
                continue
            raw_subset = df_raw_injected.iloc[[i for i in idx_list if i < len(df_raw_injected)]]
            has_synthetic = int(raw_subset["is_synthetic"].any())
            scenario = ""
            if has_synthetic:
                synth_rows = raw_subset[raw_subset["is_synthetic"] == 1]
                scenario = synth_rows["scenario_id"].iloc[0] if len(synth_rows) > 0 else ""
        else:
            # [FIX-A] Gunakan set intersection dengan wazuh_alert_ids
            alert_ids = set(wid_str.split("|"))
            has_synthetic = 0
            scenario = ""
            
            for wid in alert_ids:
                if wid in synthetic_ids:
                    info = synthetic_ids[wid]
                    if info["is_synthetic"] == 1:
                        has_synthetic = 1
                        scenario = info["scenario_id"]
                        break  # Cukup ketemu 1 synthetic
            
        ground_truth.append(has_synthetic)
        scenario_ids.append(scenario)

    df["ground_truth"] = ground_truth
    df["scenario_id"]  = scenario_ids
    n_pos = sum(ground_truth)
    log.info("[PROPAGATE] %d positif (%.1f%%) dari %d meta-alert",
             n_pos, n_pos / len(df) * 100, len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/rbta_ready_ALL.csv")
    parser.add_argument("--output", default="data/injected/rbta_injected.csv")
    parser.add_argument("--scenarios", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    args = parser.parse_args()

    run_injection(args.input, args.output, args.scenarios)