"""
retrospective_labeling.py — Ground Truth Labeling untuk Dataset Offline RBTA
=============================================================================
Metodologi:
  1. Rule-Based Filtering (rule_level >= 12 = High Confidence Attack)
  2. MITRE ATT&CK Chain Analysis (multi-tactic sequence = Planned Attack)
  3. Suspicious Payload Detection (full_log mengandung command mencurigakan)
  4. External IP Cross-Reference (srcip external = higher risk)
  5. Critical MITRE Tactic Flag (Execution, Lateral Movement, dll)

Output:
  - rbta_ready_ALL_labeled.csv (dengan kolom ground_truth_label + confidence_score)
  - labeling_report.txt (statistik dan justifikasi per kategori)
  - sample_manual_review.csv (10% sample untuk expert judgment)
"""

import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Path ──────────────────────────────────────────────────────────────────────
INPUT_CSV  = "data/rbta_ready_ALL.csv"
OUTPUT_DIR = "data"

# ── Suspicious Command Patterns (Linux/Windows) ──────────────────────────────
SUSPICIOUS_COMMANDS = [
    # Reverse shell & remote access
    r'\bnc\b', r'\bncat\b', r'\bnmap\b', r'\bbash\s+-i\b', r'\b/dev/tcp/',
    r'\bpowershell\b.*-enc', r'\bcmd\.exe\b', r'\brdesktop\b', r'\bssh\b.*-L',
    # Data exfiltration
    r'\bcurl\b.*-d', r'\bwget\b.*--post', r'\bscp\b', r'\bftp\b',
    r'\bbase64\b.*-d', r'\btar\b.*czf',
    # Privilege escalation
    r'\bsudo\b', r'\bsu\s+root\b', r'\bchmod\s+[0-7]*7', r'\bchown\b.*root',
    r'\buseradd\b', r'\bpasswd\b', r'\bvisudo\b',
    # Persistence
    r'\bcrontab\b', r'\b/etc/cron', r'\b\.bashrc\b', r'\b\.profile\b',
    r'\b/etc/passwd\b', r'\b/etc/shadow\b', r'\bauthorized_keys\b',
    r'\bsystemctl\b.*(enable|start)',
    # Defense evasion
    r'\bhistory\s+-c\b', r'\brm\b.*-rf', r'\bshred\b', r'\bunset\b.*HISTFILE\b',
    r'\biptables\b.*-F', r'\bkill\b.*-9',
    # Reconnaissance
    r'\bwhoami\b', r'\bid\b', r'\buname\s+-a\b', r'\bifconfig\b',
    r'\bip\s+addr\b', r'\bnetstat\b', r'\bps\s+-ef\b',
    r'\bcat\s+/etc/passwd\b',
]

# Pre-compile regex untuk performa
SUSPICIOUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_COMMANDS]


def detect_suspicious_payload(text: str) -> tuple[bool, list[str]]:
    """
    Deteksi apakah text (full_log/audit_command) mengandung command mencurigakan.
    Kembalikan (is_suspicious, list_matched_patterns).
    """
    if pd.isna(text) or not text:
        return False, []

    matched = []
    for i, pattern in enumerate(SUSPICIOUS_PATTERNS):
        if pattern.search(text):
            matched.append(SUSPICIOUS_COMMANDS[i])

    return len(matched) > 0, matched


def label_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beri label ground_truth_label dan confidence_score untuk setiap baris.

    Label:
      1 = Serangan nyata (attack)
      0 = Anomali operasional / noise (benign)

    Confidence Score:
      0.0 - 1.0 (berapa yakin ini serangan asli)

    Metodologi Scoring:
      - rule_level >= 12                 : +0.4 (high severity)
      - has_critical_mitre == 1          : +0.2 (taktik kritis)
      - Suspicious payload detected      : +0.2 (bukti forensik)
      - External IP (srcip_type=external): +0.1 (risk factor)
      - Multi-tactic MITRE (>=2 tactic)  : +0.1 (attack chain)

    Threshold:
      - score >= 0.6  → ground_truth_label = 1 (Attack)
      - score <  0.6  → ground_truth_label = 0 (Benign/Noise)
    """
    log.info("=== RETROSPECTIVE LABELING ===")
    log.info("Total rows: %d", len(df))

    scores = np.zeros(len(df), dtype=np.float32)
    reasons = [[] for _ in range(len(df))]

    # ── 1. Rule-Based Filtering ─────────────────────────────────────────
    mask_high_severity = df["rule_level"] >= 12
    scores[mask_high_severity] += 0.4
    for idx in df[mask_high_severity].index:
        reasons[idx].append(f"rule_level={df.loc[idx, 'rule_level']} (>=12)")

    mask_medium_severity = (df["rule_level"] >= 9) & (df["rule_level"] < 12)
    scores[mask_medium_severity] += 0.2
    for idx in df[mask_medium_severity].index:
        reasons[idx].append(f"rule_level={df.loc[idx, 'rule_level']} (9-11)")

    # ── 2. Critical MITRE Tactic ────────────────────────────────────────
    mask_critical_mitre = df["has_critical_mitre"] == 1
    scores[mask_critical_mitre] += 0.2
    for idx in df[mask_critical_mitre].index:
        reasons[idx].append("critical_mitre_tactic")

    # ── 3. Suspicious Payload Detection (sample 100K untuk performa) ─────
    log.info("Detecting suspicious payloads...")
    # Full scan jika dataset < 500K, sample jika lebih besar
    if len(df) <= 500000:
        mask_suspicious = df.apply(
            lambda row: detect_suspicious_payload(
                str(row.get("full_log", "")) + " " + str(row.get("audit_command", ""))
            )[0],
            axis=1,
        )
    else:
        log.info("  Dataset besar, sampling 200K baris untuk payload analysis...")
        sample_size = min(200000, len(df))
        sample_idx = np.random.choice(len(df), sample_size, replace=False)
        mask_suspicious = pd.Series(False, index=df.index)

        for idx in sample_idx:
            text = str(df.iloc[idx].get("full_log", "")) + " " + \
                   str(df.iloc[idx].get("audit_command", ""))
            is_susp, _ = detect_suspicious_payload(text)
            mask_suspicious.iloc[idx] = is_susp

    scores[mask_suspicious] += 0.2
    for idx in df[mask_suspicious].index:
        reasons[idx].append("suspicious_payload")

    # ── 4. External IP Cross-Reference ──────────────────────────────────
    mask_external = df["srcip_type"] == "external"
    scores[mask_external] += 0.1
    for idx in df[mask_external].index:
        reasons[idx].append("external_srcip")

    # ── 5. Multi-Tactic MITRE Chain (Attack Chain Indicator) ────────────
    mitre_tactics = df["mitre_tactic"].fillna("")
    mask_multi_tactic = mitre_tactics.apply(lambda x: len(str(x).split("|")) >= 2)
    scores[mask_multi_tactic] += 0.1
    for idx in df[mask_multi_tactic].index:
        reasons[idx].append(f"multi_tactic_chain ({mitre_tactics.iloc[idx]})")

    # ── Final Labeling ──────────────────────────────────────────────────
    df["confidence_score"] = np.clip(scores, 0.0, 1.0)
    df["ground_truth_label"] = (df["confidence_score"] >= 0.6).astype(int)
    df["label_reasons"] = ["; ".join(r) if r else "benign_operational" for r in reasons]

    # ── Statistik ───────────────────────────────────────────────────────
    n_attack = df["ground_truth_label"].sum()
    n_benign = len(df) - n_attack
    pct_attack = n_attack / len(df) * 100

    log.info("\n" + "=" * 70)
    log.info("LABELING RESULTS")
    log.info("=" * 70)
    log.info("Attack (label=1)    : %d (%.2f%%)", n_attack, pct_attack)
    log.info("Benign (label=0)    : %d (%.2f%%)", n_benign, 100 - pct_attack)
    log.info("Confidence stats:")
    log.info("  Mean : %.3f", df["confidence_score"].mean())
    log.info("  Median: %.3f", df["confidence_score"].median())
    log.info("  Max  : %.3f", df["confidence_score"].max())

    # Distribusi score
    score_bins = pd.cut(df["confidence_score"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    log.info("\nScore distribution:")
    log.info("\n%s", score_bins.value_counts().sort_index())

    return df


def generate_labeling_report(df: pd.DataFrame) -> str:
    """Generate laporan lengkap untuk validasi manual."""
    lines = []
    lines.append("=" * 70)
    lines.append("  RETROSPECTIVE LABELING REPORT")
    lines.append("=" * 70)

    lines.append("\n[1] LABEL DISTRIBUTION")
    for label, cnt in df["ground_truth_label"].value_counts().items():
        label_name = "ATTACK" if label == 1 else "BENIGN"
        lines.append(f"  {label_name} (label={label}): {cnt:,} ({cnt/len(df)*100:.1f}%)")

    lines.append("\n[2] TOP REASONS FOR ATTACK LABEL")
    attack_df = df[df["ground_truth_label"] == 1]
    if len(attack_df) > 0:
        reason_counts = {}
        for reason_str in attack_df["label_reasons"].dropna():
            for r in reason_str.split("; "):
                reason_counts[r] = reason_counts.get(r, 0) + 1
        for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {reason:<50}: {cnt:,}")

    lines.append("\n[3] ATTACK BY RULE GROUP")
    attack_df = df[df["ground_truth_label"] == 1]
    if len(attack_df) > 0:
        for grp, cnt in attack_df["rule_group_primary"].value_counts().head(10).items():
            lines.append(f"  {grp:<35}: {cnt:,}")

    lines.append("\n[4] ATTACK BY MITRE TACTIC")
    if "has_critical_mitre" in attack_df.columns:
        n_crit = attack_df["has_critical_mitre"].sum()
        lines.append(f"  Critical MITRE tactics: {n_crit:,} ({n_crit/len(attack_df)*100:.1f}% of attacks)")

    lines.append("\n[5] EXTERNAL IP INVOLVEMENT")
    external_attacks = attack_df[attack_df["srcip_type"] == "external"]
    if len(external_attacks) > 0:
        lines.append(f"  External IPs in attacks: {len(external_attacks)}")
        for ip, cnt in external_attacks["srcip"].value_counts().head(5).items():
            lines.append(f"    {ip:<20}: {cnt} alerts")
    else:
        lines.append("  No external IP involvement detected")

    lines.append("\n[6] SUSPICIOUS PAYLOAD EXAMPLES")
    payload_attacks = attack_df[attack_df["label_reasons"].str.contains("suspicious_payload", na=False)]
    if len(payload_attacks) > 0:
        for _, row in payload_attacks.head(5).iterrows():
            lines.append(f"\n  Alert ID: {row['wazuh_alert_id']}")
            lines.append(f"  Timestamp: {row['timestamp_utc']}")
            lines.append(f"  Rule: {row['rule_description'][:80]}")
            if pd.notna(row.get("full_log")) and row["full_log"]:
                lines.append(f"  Log: {str(row['full_log'])[:150]}")

    lines.append("\n[7] CONFIDENCE SCORE DISTRIBUTION (ATTACK ONLY)")
    if len(attack_df) > 0:
        lines.append(f"  Mean confidence: {attack_df['confidence_score'].mean():.3f}")
        lines.append(f"  Std dev        : {attack_df['confidence_score'].std():.3f}")
        for threshold in [0.6, 0.7, 0.8, 0.9, 1.0]:
            n_above = (attack_df["confidence_score"] >= threshold).sum()
            lines.append(f"    >= {threshold:.1f}: {n_above:,} ({n_above/len(attack_df)*100:.1f}%)")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def export_manual_review_sample(df: pd.DataFrame, sample_pct=0.10) -> pd.DataFrame:
    """
    Export 10% sample dari attack-labeled rows untuk expert judgment.
    Stratified sampling berdasarkan rule_group_primary dan confidence_score.
    """
    attack_df = df[df["ground_truth_label"] == 1].copy()

    if len(attack_df) == 0:
        log.warning("Tidak ada attack-labeled rows untuk manual review.")
        return pd.DataFrame()

    sample_size = max(100, int(len(attack_df) * sample_pct))
    log.info("Exporting %d rows (%.1f%% of attacks) for manual review...",
             sample_size, sample_pct)

    # Stratified: prioritaskan high-confidence dan diverse rule groups
    attack_df["strata"] = (
        attack_df["rule_group_primary"].astype(str) + "_" +
        pd.cut(attack_df["confidence_score"], bins=[0, 0.6, 0.8, 1.0],
               labels=["low", "med", "high"]).astype(str)
    )

    sample = attack_df.groupby("strata", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), max(5, int(sample_size * len(x) / len(attack_df)))),
            random_state=42
        )
    )

    # Kolom yang relevan untuk review manual
    review_cols = [
        "wazuh_alert_id", "timestamp_utc", "agent_name", "rule_group_primary",
        "rule_level", "rule_id", "rule_description", "srcip", "srcip_type",
        "criticality_score", "has_mitre", "has_critical_mitre", "mitre_tactic",
        "full_log", "audit_command", "confidence_score", "ground_truth_label",
        "label_reasons",
    ]
    available_cols = [c for c in review_cols if c in sample.columns]
    sample = sample[available_cols]

    # Tambah kolom untuk expert judgment
    sample["expert_label"] = None  # 0 atau 1 (diisi manual)
    sample["expert_notes"] = ""    # catatan reviewer

    return sample


def main():
    log.info("Loading dataset...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # Step 1: Labeling
    df_labeled = label_ground_truth(df)

    # Step 2: Save labeled dataset
    out_labeled = Path(OUTPUT_DIR) / "rbta_ready_ALL_labeled.csv"
    df_labeled.to_csv(out_labeled, index=False)
    log.info("\n[OK] Labeled dataset saved: %s (%d rows)", out_labeled, len(df_labeled))

    # Step 3: Generate report
    report = generate_labeling_report(df_labeled)
    report_path = Path(OUTPUT_DIR) / "labeling_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("[OK] Report saved: %s", report_path)
    print("\n" + report)

    # Step 4: Export manual review sample
    sample = export_manual_review_sample(df_labeled, sample_pct=0.10)
    if not sample.empty:
        sample_path = Path(OUTPUT_DIR) / "manual_review_sample.csv"
        sample.to_csv(sample_path, index=False)
        log.info("[OK] Manual review sample saved: %s (%d rows)", sample_path, len(sample))

    # Step 5: Summary untuk validasi
    log.info("\n" + "=" * 70)
    log.info("NEXT STEPS:")
    log.info("=" * 70)
    log.info("1. Buka manual_review_sample.csv di Excel/spreadsheet")
    log.info("2. Isi kolom 'expert_label' (0=Benign, 1=Attack) berdasarkan:")
    log.info("   - Cek rule_description dan full_log")
    log.info("   - Cross-check srcip di VirusTotal/AbuseIPDB")
    log.info("   - Validasi ke sistem asli jika memungkinkan")
    log.info("3. Setelah selesai, jalankan evaluation_metrics.py")
    log.info("   untuk hitung Precision, Recall, F1 vs ground truth")


if __name__ == "__main__":
    main()
