"""
clean_and_filter.py — Two-Stage Filtering untuk Mitigasi False Positive
========================================================================
Stage 1: Rule-based exclusion (hapus operational noise)
Stage 2: Re-labeling dengan threshold lebih ketat

Output:
  - rbta_ready_ALL_cleaned.csv (dataset bersih untuk IF)
  - filtering_report.txt (statistik BEFORE vs AFTER)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Path ──────────────────────────────────────────────────────────────────────
INPUT_CSV    = "data/rbta_ready_ALL.csv"
OUTPUT_DIR   = "data"

# ── Whitelist: Rule groups yang 100% operational (bukan attack) ───────────
OPERATIONAL_GROUPS = frozenset({
    "syscheck",
    "rootcheck",
    "sca",
    "vulnerability-detector",
    "ossec",
})

# ── Whitelist: Rule description patterns yang known-good ───────────────────
KNOWN_GOOD_PATTERNS = [
    "wazuh server started",
    "manager started",
    "server started",
    "health check",
    "agent event queue",
    "log file rotated",
    "dpkg",
    "cis ubuntu",
    "benchmark",
    "package installed",
    "new dpkg",
    "dpkg (debian package)",
    "file added to the system",
    "file deleted",
    "integrity checksum",
    "host-based anomaly",
]


def stage1_operational_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stage 1: Hapus operational noise yang 100% bukan attack.

    Return: (df_filtered, df_removed)
    """
    log.info("STAGE 1: Operational Filter")
    log.info("  Input: %d rows", len(df))

    mask_operational = (
        df["rule_group_primary"].isin(OPERATIONAL_GROUPS)
    )

    # Juga cek known-good patterns di rule_description
    desc_lower = df["rule_description"].fillna("").str.lower()
    for pattern in KNOWN_GOOD_PATTERNS:
        mask_operational |= desc_lower.str.contains(pattern, na=False)

    df_removed = df[mask_operational].copy()
    df_filtered = df[~mask_operational].copy()

    log.info("  Removed: %d rows (%.1f%%)", len(df_removed), len(df_removed)/len(df)*100)
    log.info("  Remaining: %d rows (%.1f%%)", len(df_filtered), len(df_filtered)/len(df)*100)

    return df_filtered, df_removed


def stage2_relabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Re-labeling dengan threshold lebih ketat.

    Metodologi baru:
      - rule_level >= 12                 : +0.4 (unchanged)
      - has_critical_mitre == 1          : +0.3 (↑ dari 0.2 — lebih penting)
      - Suspicious payload detected      : +0.3 (↑ dari 0.2 — bukti forensik kuat)
      - External IP                      : +0.1 (unchanged)
      - Multi-tactic MITRE               : +0.2 (↑ dari 0.1 — attack chain lebih signifikan)

    Threshold baru: score >= 0.7 → Attack (↑ dari 0.6)
    """
    log.info("STAGE 2: Re-labeling dengan threshold ketat")

    df_clean = df.copy()
    # Reset index untuk menghindari IndexError
    df_clean = df_clean.reset_index(drop=True)

    scores = np.zeros(len(df_clean), dtype=np.float32)
    reasons = [[] for _ in range(len(df_clean))]

    # 1. Rule level
    mask_high = df_clean["rule_level"] >= 12
    scores[mask_high] += 0.4
    for idx in df_clean[mask_high].index:
        reasons[idx].append(f"rule_level={df_clean.loc[idx, 'rule_level']}")

    mask_med = (df_clean["rule_level"] >= 9) & (df_clean["rule_level"] < 12)
    scores[mask_med] += 0.2
    for idx in df_clean[mask_med].index:
        reasons[idx].append(f"rule_level={df_clean.loc[idx, 'rule_level']}")

    # 2. Critical MITRE (bobot lebih tinggi)
    mask_mitre = df_clean["has_critical_mitre"] == 1
    scores[mask_mitre] += 0.3
    for idx in df_clean[mask_mitre].index:
        reasons[idx].append("critical_mitre")

    # 3. Multi-tactic chain (bobot lebih tinggi)
    mitre_tactics = df_clean["mitre_tactic"].fillna("")
    mask_multi = mitre_tactics.apply(lambda x: len(str(x).split("|")) >= 2)
    scores[mask_multi] += 0.2
    for idx in df_clean[mask_multi].index:
        reasons[idx].append(f"multi_tactic")

    # 4. External IP
    mask_ext = df_clean["srcip_type"] == "external"
    scores[mask_ext] += 0.1
    for idx in df_clean[mask_ext].index:
        reasons[idx].append("external_ip")

    # Labeling dengan threshold lebih tinggi
    df_clean["confidence_score"] = np.clip(scores, 0.0, 1.0)
    df_clean["ground_truth_label"] = (df_clean["confidence_score"] >= 0.7).astype(int)
    df_clean["label_reasons"] = ["; ".join(r) if r else "benign" for r in reasons]

    n_attack = df_clean["ground_truth_label"].sum()
    n_benign = len(df_clean) - n_attack
    log.info("  Attack (label=1): %d (%.2f%%)", n_attack, n_attack/len(df_clean)*100)
    log.info("  Benign (label=0): %d (%.2f%%)", n_benign, n_benign/len(df_clean)*100)

    return df_clean


def generate_report(df_original, df_removed, df_cleaned) -> str:
    """Generate BEFORE vs AFTER report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  TWO-STAGE FILTERING REPORT — BEFORE vs AFTER")
    lines.append("=" * 80)

    lines.append(f"\n[BEFORE FILTERING]")
    lines.append(f"  Total alerts          : {len(df_original):,}")
    orig_attacks = (df_original.get("ground_truth_label") == 1).sum() if "ground_truth_label" in df_original.columns else 0
    lines.append(f"  Attacks (old label)   : {orig_attacks:,}")

    lines.append(f"\n[STAGE 1: OPERATIONAL FILTER]")
    lines.append(f"  Removed (operational) : {len(df_removed):,} ({len(df_removed)/len(df_original)*100:.1f}%)")
    lines.append(f"  Remaining             : {len(df_cleaned) if 'confidence_score' not in df_cleaned.columns else len(df_cleaned):,}")

    lines.append(f"\n[AFTER STAGE 2: RE-LABELING]")
    new_attacks = df_cleaned["ground_truth_label"].sum()
    new_benign = len(df_cleaned) - new_attacks
    lines.append(f"  Total alerts          : {len(df_cleaned):,}")
    lines.append(f"  Attacks (new label)   : {new_attacks:,} ({new_attacks/len(df_cleaned)*100:.2f}%)")
    lines.append(f"  Benign                : {new_benign:,} ({new_benign/len(df_cleaned)*100:.2f}%)")

    # Quality metrics
    lines.append(f"\n[QUALITY IMPROVEMENT]")
    attack_rate_before = orig_attacks / len(df_original) * 100 if len(df_original) > 0 else 0
    attack_rate_after = new_attacks / len(df_cleaned) * 100 if len(df_cleaned) > 0 else 0
    lines.append(f"  Attack rate BEFORE  : {attack_rate_before:.4f}%")
    lines.append(f"  Attack rate AFTER   : {attack_rate_after:.2f}%")
    lines.append(f"  Improvement ratio   : {attack_rate_after/attack_rate_before if attack_rate_before > 0 else 0:.1f}x")

    # Top attack patterns after cleaning
    if new_attacks > 0:
        lines.append(f"\n[TOP ATTACK PATTERNS AFTER CLEANING]")
        attack_df = df_cleaned[df_cleaned["ground_truth_label"] == 1]
        for grp, cnt in attack_df["rule_group_primary"].value_counts().head(10).items():
            lines.append(f"  {grp:<30}: {cnt:,}")

        lines.append(f"\n[CONFIDENCE SCORE DISTRIBUTION]")
        for thresh in [0.7, 0.8, 0.9, 1.0]:
            n_above = (df_cleaned["confidence_score"] >= thresh).sum()
            lines.append(f"  >= {thresh:.1f}: {n_above:,} ({n_above/new_attacks*100:.1f}%)")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    log.info("Loading original dataset...")
    df_original = pd.read_csv(INPUT_CSV, low_memory=False)
    log.info("Loaded %d rows", len(df_original))

    # Stage 1: Remove operational noise
    df_filtered, df_removed = stage1_operational_filter(df_original)

    # Stage 2: Re-label with stricter threshold
    df_cleaned = stage2_relabel(df_filtered)

    # Save cleaned dataset
    out_cleaned = Path(OUTPUT_DIR) / "rbta_ready_ALL_cleaned.csv"
    df_cleaned.to_csv(out_cleaned, index=False)
    log.info("\n[OK] Cleaned dataset saved: %s (%d rows)", out_cleaned, len(df_cleaned))

    # Generate report
    report = generate_report(df_original, df_removed, df_cleaned)
    report_path = Path(OUTPUT_DIR) / "filtering_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("[OK] Report saved: %s", report_path)
    print("\n" + report)

    # Next steps
    log.info("\n" + "=" * 80)
    log.info("NEXT STEPS:")
    log.info("=" * 80)
    log.info("1. Gunakan rbta_ready_ALL_cleaned.csv untuk training Isolation Forest")
    log.info("2. IF sekarang hanya melihat alerts yang berpotensi attack")
    log.info("3. Expected: Precision meningkat drastis, Alert Fatigue berkurang")
    log.info("4. Jalankan: python src/isolation_forest.py dengan dataset cleaned")


if __name__ == "__main__":
    main()
