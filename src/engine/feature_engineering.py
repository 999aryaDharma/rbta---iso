"""
feature_engineering.py  —  Behavioral Feature Engineering (f10-f13)
====================================================================
Modul ini menghitung 4 fitur tambahan yang tidak bisa dihitung dalam
satu pass RBTA karena membutuhkan konteks lintas-bucket atau lintas-agent.

Fitur baru:
  f10  rule_group_entropy      -- keberagaman rule_group dalam bucket per-agent
                                  Shannon entropy dari distribusi rule_group.
                                  Rendah = serangan monoton (brute-force).
                                  Tinggi = multi-stage (APT candidate).

  f11  tactic_progression_score -- urutan taktik MITRE mengikuti kill chain.
                                  0.0 = tidak ada MITRE atau acak.
                                  1.0 = semua taktik berurutan maju di kill chain.

  f12  deviation_from_baseline  -- deviasi alert_count dari rolling average
                                  24 jam terakhir per agent.
                                  Nilai positif = lebih tinggi dari biasanya.
                                  Negatif = lebih rendah (mungkin evasion).

  f13  cross_agent_spread       -- jumlah agent unik yang sama srcip-nya
                                  dalam window 1 jam. Proxy untuk lateral
                                  movement dari satu sumber IP.

Cara penggunaan:
    from src.feature_engineering import enrich_features
    df_meta = enrich_features(df_meta)

Semua fungsi mengembalikan DataFrame baru (tidak mutasi input).
Kompleksitas dinotasikan per fungsi.
"""

import logging
import json
from math import log2
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Kill chain ordering ───────────────────────────────────────────────────────
# Sumber: MITRE ATT&CK Enterprise v14 — urutan fase dalam serangan APT tipikal.
# WAJIB didokumentasikan sebagai "Tabel Kill Chain Ordering" di Bab 3.
# Dataset INSTIKI mungkin hanya memiliki subset taktik ini.

KILL_CHAIN_ORDER: dict[str, int] = {
    "Reconnaissance":         1,
    "Resource Development":   2,
    "Initial Access":         3,
    "Execution":              4,
    "Persistence":            5,
    "Privilege Escalation":   6,
    "Defense Evasion":        7,
    "Credential Access":      8,
    "Discovery":              9,
    "Lateral Movement":       10,
    "Collection":             11,
    "Command and Control":    12,
    "Exfiltration":           13,
    "Impact":                 14,
}

# Window default untuk f12 dan f13
BASELINE_WINDOW_HOURS = 24
SPREAD_WINDOW_HOURS   = 1


# ══════════════════════════════════════════════════════════════════════════════
# f10 — Rule Group Entropy
# ══════════════════════════════════════════════════════════════════════════════

def compute_rule_group_entropy(rule_group_counts: dict[str, int]) -> float:
    """
    Hitung Shannon entropy dari distribusi rule_group dalam satu bucket.

    Dipanggil per baris df_meta. Input adalah dict hasil JSON parse dari
    kolom rule_id_dist atau kolom khusus rule_group_dist yang dihasilkan
    oleh rbta_algorithm_02 v5.

    Kompleksitas: O(k) dimana k = jumlah rule_group unik dalam bucket.
    Nilai maksimum: log2(k) jika distribusi seragam.

    Parameters
    ----------
    rule_group_counts : dict { rule_group: count }

    Returns
    -------
    float — entropy dalam bit. 0.0 jika hanya satu grup atau dict kosong.
    """
    if not rule_group_counts:
        return 0.0

    total = sum(rule_group_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in rule_group_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * log2(p)

    return round(entropy, 4)


def add_rule_group_entropy(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan kolom rule_group_entropy (f10) ke df_meta.

    Menggunakan kolom 'rule_group_dist' jika ada (output rbta v5),
    fallback ke rule_id_dist sebagai proxy jika tidak ada.

    Kompleksitas total: O(n * k)
    """
    df = df_meta.copy()

    if "rule_group_dist" in df.columns:
        source_col = "rule_group_dist"
    elif "rule_id_dist" in df.columns:
        source_col = "rule_id_dist"
        log.warning(
            "Kolom rule_group_dist tidak ditemukan. "
            "Menggunakan rule_id_dist sebagai proxy untuk entropy. "
            "Pastikan rbta_algorithm_02 v5 yang menghasilkan CSV ini."
        )
    else:
        log.warning("Tidak ada kolom distribusi yang tersedia — rule_group_entropy di-set 0.")
        df["rule_group_entropy"] = 0.0
        return df

    def _safe_entropy(val: object) -> float:
        if pd.isna(val) or val == "":
            return 0.0
        try:
            if isinstance(val, str):
                d = json.loads(val)
            elif isinstance(val, dict):
                d = val
            else:
                return 0.0
            return compute_rule_group_entropy(d)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    df["rule_group_entropy"] = df[source_col].apply(_safe_entropy)
    log.info(
        "f10 rule_group_entropy: min=%.3f  median=%.3f  max=%.3f",
        df["rule_group_entropy"].min(),
        df["rule_group_entropy"].median(),
        df["rule_group_entropy"].max(),
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# f11 — Tactic Progression Score
# ══════════════════════════════════════════════════════════════════════════════

def tactic_progression_score(tactics_ordered: list[str]) -> float:
    """
    Skor seberapa "maju" urutan taktik MITRE mengikuti kill chain.

    Algoritma:
      1. Filter taktik yang dikenal di KILL_CHAIN_ORDER.
      2. Konversi ke urutan numerik kill chain.
      3. Hitung fraksi pasangan berurutan yang "maju" (b > a).

    Kompleksitas: O(m) dimana m = len(tactics_ordered)

    Returns
    -------
    float:
      0.0  -- tidak ada MITRE, atau semua taktik sama, atau mundur
      1.0  -- semua pasangan berurutan maju di kill chain
    """
    if not tactics_ordered:
        return 0.0

    ordered_nums = [
        KILL_CHAIN_ORDER[t]
        for t in tactics_ordered
        if t in KILL_CHAIN_ORDER
    ]

    if len(ordered_nums) < 2:
        return 0.0

    pairs = [
        (ordered_nums[i], ordered_nums[i + 1])
        for i in range(len(ordered_nums) - 1)
    ]
    n_forward = sum(1 for a, b in pairs if b > a)
    return round(n_forward / len(pairs), 4)


def add_tactic_progression(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan kolom tactic_progression_score (f11) ke df_meta.

    Membutuhkan kolom 'mitre_tactic' yang berisi string taktik
    dipisahkan pipe (|), misalnya "Execution|Lateral Movement".
    Ini berasal dari Layer 2 json_orches.py yang tersimpan di full_row.

    Jika kolom tidak ada atau kosong, nilai di-set 0.0.
    Kompleksitas total: O(n * m)
    """
    df = df_meta.copy()

    if "mitre_tactic" not in df.columns:
        log.warning(
            "Kolom mitre_tactic tidak ditemukan — tactic_progression_score di-set 0. "
            "Kolom ini ada di Layer 2 (full_row) dari json_orches.py. "
            "Pastikan df_meta berasal dari CSV full, bukan CSV minimal."
        )
        df["tactic_progression_score"] = 0.0
        return df

    def _parse_and_score(val: object) -> float:
        if pd.isna(val) or val == "":
            return 0.0
        tactics = [t.strip() for t in str(val).split("|") if t.strip()]
        return tactic_progression_score(tactics)

    df["tactic_progression_score"] = df["mitre_tactic"].apply(_parse_and_score)

    # Peringatan jika semua nilai 0 (MITRE coverage sangat rendah)
    n_nonzero = (df["tactic_progression_score"] > 0).sum()
    coverage  = n_nonzero / len(df) * 100
    if coverage < 1.0:
        log.warning(
            "f11 tactic_progression_score: hanya %.1f%% baris nonzero. "
            "MITRE coverage sangat rendah — fitur ini mungkin tidak berkontribusi "
            "signifikan pada Isolation Forest. Dokumentasikan sebagai limitation.",
            coverage,
        )
    else:
        log.info(
            "f11 tactic_progression_score: %.1f%% baris memiliki sinyal MITRE.",
            coverage,
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# f12 — Deviation from Baseline
# ══════════════════════════════════════════════════════════════════════════════

def compute_deviation_from_baseline(
    df_meta:      pd.DataFrame,
    window_hours: int = BASELINE_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    Tambahkan kolom deviation_from_baseline (f12) ke df_meta.

    Untuk setiap baris, hitung rata-rata alert_count dari baris lain
    yang sama agent_name-nya dalam window [t - window_hours, t).
    Deviasi = (current - baseline_mean) / baseline_mean.

    Nilai positif  = lebih tinggi dari biasanya (anomali burst)
    Nilai negatif  = lebih rendah dari biasanya (mungkin evasion)
    Nilai 0        = tidak ada data historis atau sesuai rata-rata

    Kompleksitas: O(n * n) naif — cukup untuk batch CSV <10k baris.
    Untuk dataset lebih besar, gunakan groupby + rolling.

    Parameters
    ----------
    df_meta      : DataFrame meta-alert dengan kolom start_time, agent_name,
                   alert_count.
    window_hours : lebar window baseline dalam jam.

    Returns
    -------
    pd.DataFrame dengan kolom deviation_from_baseline (float, clipped ke [-3, 3]).
    """
    df = df_meta.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    n = len(df)
    if n == 0:
        df["deviation_from_baseline"] = 0.0
        return df

    df_sorted   = df.sort_values("start_time").reset_index(drop=True)
    deviations  = np.zeros(n, dtype=float)
    window_td   = pd.Timedelta(hours=window_hours)

    for i in range(n):
        t_now   = df_sorted.at[i, "start_time"]
        agent   = df_sorted.at[i, "agent_name"]
        t_from  = t_now - window_td

        mask_baseline = (
            (df_sorted["agent_name"] == agent) &
            (df_sorted["start_time"] >= t_from) &
            (df_sorted["start_time"] < t_now)
        )
        baseline_rows = df_sorted.loc[mask_baseline, "alert_count"]

        if len(baseline_rows) == 0:
            deviations[i] = 0.0
        else:
            baseline_mean = baseline_rows.mean()
            if baseline_mean > 0:
                deviations[i] = (
                    df_sorted.at[i, "alert_count"] - baseline_mean
                ) / baseline_mean
            else:
                deviations[i] = 0.0

    # Clip ke [-3, 3] — deviasi 3x dari baseline sudah sangat signifikan
    df_sorted["deviation_from_baseline"] = np.clip(deviations, -3.0, 3.0).round(4)

    # Kembalikan ke urutan original df
    df_sorted = df_sorted.set_index(df.sort_values("start_time").index)
    df["deviation_from_baseline"] = df_sorted["deviation_from_baseline"].values

    log.info(
        "f12 deviation_from_baseline: min=%.3f  median=%.3f  max=%.3f  "
        "(window=%dh, clipped to [-3, 3])",
        df["deviation_from_baseline"].min(),
        df["deviation_from_baseline"].median(),
        df["deviation_from_baseline"].max(),
        window_hours,
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# f13 — Cross-Agent Spread
# ══════════════════════════════════════════════════════════════════════════════

def compute_cross_agent_spread(
    df_meta:      pd.DataFrame,
    window_hours: int = SPREAD_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    Tambahkan kolom cross_agent_spread (f13) ke df_meta.

    Untuk setiap baris, cari semua IP di attacker_ips-nya.
    Kemudian hitung berapa banyak agent unik yang muncul bersama
    IP yang sama dalam window ±window_hours.

    Nilai tinggi = IP yang sama menyerang banyak agent → lateral movement.
    Nilai 0      = tidak ada IP eksternal atau IP hanya menyerang satu agent.

    Kompleksitas: O(n * n * k) dimana k = jumlah attacker IP per baris.
    Untuk dataset besar, buat index IP → rows terlebih dahulu.

    Parameters
    ----------
    df_meta      : DataFrame meta-alert dengan kolom start_time, agent_name,
                   attacker_ips (pipe-separated string).
    window_hours : lebar window dalam jam (symmetric: ±window_hours).

    Returns
    -------
    pd.DataFrame dengan kolom cross_agent_spread (int).
    """
    df = df_meta.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    n = len(df)
    if n == 0:
        df["cross_agent_spread"] = 0
        return df

    # Precompute: index dari setiap IP ke baris yang memilikinya
    # Key: ip_str, Value: list of row index
    ip_to_rows: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        raw_ips = str(row.get("attacker_ips", "") or "")
        ips = [ip.strip() for ip in raw_ips.split("|") if ip.strip()]
        for ip in ips:
            if ip not in ip_to_rows:
                ip_to_rows[ip] = []
            ip_to_rows[ip].append(idx)

    if not ip_to_rows:
        log.info("f13 cross_agent_spread: tidak ada IP eksternal ditemukan — semua nilai 0.")
        df["cross_agent_spread"] = 0
        return df

    window_td = pd.Timedelta(hours=window_hours)
    spreads   = np.zeros(n, dtype=int)

    for i, (idx, row) in enumerate(df.iterrows()):
        raw_ips = str(row.get("attacker_ips", "") or "")
        ips     = [ip.strip() for ip in raw_ips.split("|") if ip.strip()]
        if not ips:
            spreads[i] = 0
            continue

        t_now  = row["start_time"]
        t_from = t_now - window_td
        t_to   = t_now + window_td
        max_spread = 0

        for ip in ips:
            candidate_rows = ip_to_rows.get(ip, [])
            agents_in_window: set[str] = set()

            for cand_idx in candidate_rows:
                cand_time  = df.at[cand_idx, "start_time"]
                cand_agent = df.at[cand_idx, "agent_name"]
                if t_from <= cand_time <= t_to:
                    agents_in_window.add(cand_agent)

            max_spread = max(max_spread, len(agents_in_window))

        spreads[i] = max_spread

    df["cross_agent_spread"] = spreads

    n_multi = (df["cross_agent_spread"] > 1).sum()
    log.info(
        "f13 cross_agent_spread: %d baris (%.1f%%) memiliki spread > 1 agent "
        "(kandidat lateral movement, window=±%dh)",
        n_multi, n_multi / len(df) * 100, window_hours,
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline: enrich_features — terapkan f10-f13 sekaligus
# ══════════════════════════════════════════════════════════════════════════════

def enrich_features(
    df_meta:               pd.DataFrame,
    baseline_window_hours: int = BASELINE_WINDOW_HOURS,
    spread_window_hours:   int = SPREAD_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    Terapkan semua 4 fitur behavioral (f10-f13) ke df_meta secara berurutan.

    Urutan eksekusi:
      1. add_rule_group_entropy    → f10 (tidak bergantung pada baris lain)
      2. add_tactic_progression    → f11 (tidak bergantung pada baris lain)
      3. compute_deviation_from_baseline → f12 (bergantung pada urutan waktu)
      4. compute_cross_agent_spread      → f13 (bergantung pada IP lintas baris)

    f12 dan f13 paling lambat karena O(n²) — untuk >10k baris pertimbangkan
    optimasi dengan rolling groupby dan IP index.

    Parameters
    ----------
    df_meta               : DataFrame hasil run_rbta() yang sudah melalui
                            add_if_features() dari isolation_forest.py.
    baseline_window_hours : window untuk f12 (default 24 jam).
    spread_window_hours   : window untuk f13 (default 1 jam).

    Returns
    -------
    pd.DataFrame dengan kolom tambahan:
      rule_group_entropy, tactic_progression_score,
      deviation_from_baseline, cross_agent_spread
    """
    import time
    t0 = time.perf_counter()

    log.info("[ENRICH] Memulai feature engineering behavioral (f10-f13) ...")

    df = add_rule_group_entropy(df_meta)
    log.info("[ENRICH] f10 selesai.")

    df = add_tactic_progression(df)
    log.info("[ENRICH] f11 selesai.")

    df = compute_deviation_from_baseline(df, window_hours=baseline_window_hours)
    log.info("[ENRICH] f12 selesai.")

    df = compute_cross_agent_spread(df, window_hours=spread_window_hours)
    log.info("[ENRICH] f13 selesai.")

    elapsed = (time.perf_counter() - t0) * 1000
    log.info("[ENRICH] Semua fitur behavioral selesai dalam %.1f ms.", elapsed)

    # Pastikan semua kolom baru bertipe numerik
    for col, dtype in {
        "rule_group_entropy":       float,
        "tactic_progression_score": float,
        "deviation_from_baseline":  float,
        "cross_agent_spread":       int,
    }.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)

    _print_enrichment_summary(df)
    return df


def _print_enrichment_summary(df: pd.DataFrame) -> None:
    new_cols = [
        "rule_group_entropy",
        "tactic_progression_score",
        "deviation_from_baseline",
        "cross_agent_spread",
    ]
    present = [c for c in new_cols if c in df.columns]
    if not present:
        return
    log.info("\n[ENRICH SUMMARY]\n%s", df[present].describe().round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Standalone entry point — untuk pengujian modul secara independen
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/meta_alerts_rbta.csv"
    log.info("Loading: %s", csv_path)

    df = pd.read_csv(csv_path)
    df = enrich_features(df)

    out_path = csv_path.replace(".csv", "_enriched.csv")
    df.to_csv(out_path, index=False)
    log.info("Saved enriched CSV: %s", out_path)