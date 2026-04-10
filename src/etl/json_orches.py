"""
json_orches.py  —  ETL Pipeline: Wazuh JSON → rbta_ready_ALL.csv
=================================================================
Revisi dari versi sebelumnya:

  [FIX-1] MITRE ATT&CK dipindahkan ke Layer 1 (minimal_row)
      Versi lama menaruh mitre di Layer 2 sehingga RBTA Engine buta
      terhadap konteks taktik serangan. Kini has_mitre dan
      has_critical_mitre tersedia di Layer 1 sebagai flag integer (0/1)
      sehingga RBTA dapat mengakumulasi mitre_hit_count per window
      dan Isolation Forest mendapat sinyal taktik MITRE secara langsung.

  [FIX-2] criticality_score diubah skala menjadi 1-4 (bukan 0-4)
      Nilai "unknown" di-default ke 1 (low) agar konsisten dengan
      preprocessing_01.py dan rbta_algorithm_02.py.

  [DOCS] AGENT_CRITICALITY didokumentasikan sebagai domain assumption
      Tabel ini spesifik untuk UPT TIK INSTIKI dan wajib dicantumkan
      sebagai "Tabel Pemetaan Aset Kritis" di Bab 3 skripsi.

Arsitektur Layer:
  Layer 1 (minimal_row) — RBTA core + MITRE flags (9 kolom)
  Layer 2 (full_row)    — enrichment ML + forensik
"""

import json
import logging
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Path default ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\KAMPUS\SEMINAR\data")
OUT_FULL    = BASE_DIR / "rbta_ready_v3.csv"
OUT_MINIMAL = BASE_DIR / "rbta_raw_minimal.csv"
OUT_REPORT  = BASE_DIR / "data_quality_report.txt"

# ── Dedup key ─────────────────────────────────────────────────────────────────
DEDUP_KEY = "wazuh_alert_id"

# ── Group priority: severity-weight berbasis data ─────────────────────────────
# Dibuat dari rata-rata rule.level per group di dataset INSTIKI.
# WAJIB didokumentasikan sebagai "Tabel Semantic Group Ordering" di Bab 3.
GROUP_SEVERITY_WEIGHT: dict[str, int] = {
    "attack":                  10,
    "sql_injection":           10,
    "authentication_failed":    9,
    "access_control":           8,
    "pam":                      7,
    "web":                      7,
    "virus":                    7,
    "nginx":                    6,
    "audit":                    6,
    "clamd":                    6,
    "accesslog":                5,
    "system_error":             5,
    "audit_selinux":            5,
    "syscheck":                 5,
    "syscheck_file":            5,
    "syscheck_entry_modified":  5,
    "syscheck_entry_added":     4,
    "authentication_success":   3,
    "freshclam":                3,
    "syslog":                   3,
    "rootcheck":                2,
    "ossec":                    1,
}


def pick_primary_group(groups: list[str]) -> str:
    """
    Pilih grup dengan severity weight tertinggi dari list rule.groups.
    Jika seri, pilih yang pertama muncul (urutan Wazuh).
    """
    if not groups:
        return "unknown"
    return max(groups, key=lambda g: GROUP_SEVERITY_WEIGHT.get(g, 0))


# ── Agent criticality ─────────────────────────────────────────────────────────
# DOMAIN ASSUMPTION — hanya valid di lingkungan UPT TIK INSTIKI.
# Skala: 1=Low, 2=Medium, 3=High, 4=Critical
# Sumber: fungsi server dalam infrastruktur produksi UPT TIK INSTIKI.
AGENT_CRITICALITY: dict[str, str] = {
    "soc-1":         "low",
    "pusatkarir":    "high",
    "dfir-iris":     "critical",
    "siput":         "medium",
    "proxy-manager": "high",
    "e-kuesioner":   "medium",
    "sads":          "high",
    "DVWA":          "low",
}

CRITICALITY_SCORE: dict[str, int] = {
    "critical": 4,
    "high":     3,
    "medium":   2,
    "low":      1,
    "unknown":  1,  # [FIX-2] default 1, bukan 0
}

# Taktik MITRE yang dianggap kritis — wajib justifikasi tabel di Bab 3
CRITICAL_MITRE_TACTICS: frozenset[str] = frozenset({
    "Execution",
    "Lateral Movement",
    "Credential Access",
    "Exfiltration",
    "Privilege Escalation",
    "Defense Evasion",
})

# ── srcip handling ────────────────────────────────────────────────────────────
INTERNAL_PREFIXES: tuple[str, ...] = (
    "10.", "172.16.", "172.17.", "192.168.", "127.",
)


def classify_srcip(ip: Optional[str]) -> tuple[Optional[str], str]:
    """
    Kembalikan (raw_srcip, srcip_type) tanpa asumsi IP = attacker.

    srcip_type:
      'internal' -- RFC1918 atau loopback (proxy, NAT, atau internal client)
      'external' -- IP publik (kandidat attacker, bukan kepastian)
      'none'     -- field tidak ada (HIDS event: syscheck, rootcheck, audit)
    """
    if not ip or pd.isna(ip):
        return None, "none"
    if any(ip.startswith(p) for p in INTERNAL_PREFIXES):
        return ip, "internal"
    return ip, "external"


# ── Verbose error tracking ────────────────────────────────────────────────────

class ParseStats:
    """Pelacak statistik parsing verbose per failure mode."""

    def __init__(self) -> None:
        self.total:    int = 0
        self.success:  int = 0
        self.failures: dict[str, list[int]] = defaultdict(list)

    def fail(self, lineno: int, reason: str) -> None:
        self.failures[reason].append(lineno)

    def report(self) -> None:
        total_fail = sum(len(v) for v in self.failures.values())
        log.info(
            "Parse stats — total=%d  success=%d  fail=%d",
            self.total, self.success, total_fail,
        )
        for reason, lines in self.failures.items():
            log.warning("  failure[%s] = %d baris", reason, len(lines))
            if len(lines) <= 3:
                log.debug("    baris: %s", lines)


# ── Parser utama ──────────────────────────────────────────────────────────────

def parse_jsonl(
    filepath: Path,
) -> tuple[list[dict], list[dict], ParseStats]:
    """
    Parse JSONL Wazuh. Kembalikan:
      records_full    : Layer 1 + Layer 2 (untuk rbta_ready_ALL.csv)
      records_minimal : Layer 1 saja (untuk rbta_raw_minimal.csv)
      stats           : ParseStats verbose
    """
    stats = ParseStats()
    records_full:    list[dict] = []
    records_minimal: list[dict] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stats.total += 1
            line = line.strip()

            if not line:
                stats.fail(lineno, "baris_kosong")
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                stats.fail(lineno, f"json_decode_error: {exc}")
                continue

            ts_raw = rec.get("timestamp", "")
            if not ts_raw:
                stats.fail(lineno, "timestamp_kosong")
                continue
            try:
                ts = pd.to_datetime(ts_raw, utc=True)
            except Exception as exc:
                stats.fail(lineno, f"timestamp_parse_error: {exc}")
                continue

            agent = rec.get("agent", {})
            rule  = rec.get("rule", {})
            if not rule.get("id"):
                stats.fail(lineno, "rule_id_kosong")
                continue
            if not rule.get("groups"):
                stats.fail(lineno, "rule_groups_kosong")
                continue

            data     = rec.get("data", {})
            mitre    = rule.get("mitre", {})
            syscheck = rec.get("syscheck", {})
            decoder  = rec.get("decoder", {})
            audit    = data.get("audit", {})

            agent_name = str(agent.get("name", "unknown"))
            groups     = rule.get("groups", [])
            raw_srcip  = data.get("srcip", None)
            srcip, srcip_type = classify_srcip(raw_srcip)
            wazuh_id   = str(rec.get("id", ""))

            criticality_label = AGENT_CRITICALITY.get(agent_name, "unknown")
            criticality_score = CRITICALITY_SCORE[criticality_label]

            # ── [FIX-1] MITRE ATT&CK dipindah ke Layer 1 ─────────────────
            mitre_tactics: list[str] = mitre.get("tactic", [])
            has_mitre          = 1 if len(mitre_tactics) > 0 else 0
            has_critical_mitre = 1 if any(
                t in CRITICAL_MITRE_TACTICS for t in mitre_tactics
            ) else 0
            # ─────────────────────────────────────────────────────────────

            # ── Layer 1: minimal — RBTA core (9 kolom) ───────────────────
            minimal_row: dict = {
                "wazuh_alert_id":      wazuh_id,
                "timestamp_utc":       ts,
                "agent_id":            str(agent.get("id", "000")).zfill(3),
                "agent_name":          agent_name,
                "rule_group_primary":  pick_primary_group(groups),
                "rule_level":          int(rule.get("level", 0)),
                "rule_id":             str(rule.get("id", "0")),
                "srcip":               srcip,
                "srcip_type":          srcip_type,
                "criticality_score":   criticality_score,
                # [FIX-1] MITRE flags — wajib untuk akumulasi RBTA & IF
                "has_mitre":           has_mitre,
                "has_critical_mitre":  has_critical_mitre,
            }
            records_minimal.append(minimal_row)

            # ── Layer 2: enriched — ML + forensik ────────────────────────
            full_row: dict = {
                **minimal_row,
                "agent_criticality":   criticality_label,
                "mitre_tactic":        "|".join(mitre_tactics),
                "mitre_technique":     "|".join(mitre.get("technique", [])),
                "mitre_id":            "|".join(mitre.get("id", [])),
                "syscheck_event":      syscheck.get("event", ""),
                "syscheck_path":       syscheck.get("path", ""),
                "syscheck_mode":       syscheck.get("mode", ""),
                "rule_description":    str(rule.get("description", "")),
                "rule_groups_all":     json.dumps(groups),
                "rule_firedtimes":     int(rule.get("firedtimes", 1)),
                "agent_ip":            str(agent.get("ip", "")),
                "decoder_name":        str(decoder.get("name", "")),
                "location":            str(rec.get("location", "")),
                "data_url":            str(data.get("url", "")),
                "audit_command":       str(audit.get("command", "")),
                "manager_name":        str(rec.get("manager", {}).get("name", "")),
                "full_log":            str(rec.get("full_log", ""))[:300],
            }
            records_full.append(full_row)
            stats.success += 1

    return records_full, records_minimal, stats


# ── Build DataFrame ───────────────────────────────────────────────────────────

def build_df(records: list[dict], dedup_key: str) -> pd.DataFrame:
    """Bangun DataFrame dari records, dedup berdasarkan dedup_key, sort temporal."""
    if not records:
        log.warning("build_df dipanggil dengan records kosong.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    before = len(df)
    df = df.drop_duplicates(subset=[dedup_key])
    after  = len(df)
    log.info(
        "Dedup key='%s': %d duplikat dihapus → %d baris tersisa",
        dedup_key, before - after, after,
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["rule_level"] = df["rule_level"].astype(int)
    return df


# ── Data quality report ───────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> str:
    """Hasilkan teks laporan kualitas data untuk logging dan file."""
    lines: list[str] = []
    lines.append("=" * 65)
    lines.append("  DATA QUALITY REPORT — rbta_ready_ALL.csv")
    lines.append("=" * 65)

    lines.append("\n[ MISSING VALUES ]")
    for col, cnt in df.isna().sum().items():
        pct  = cnt / len(df) * 100
        flag = " !" if pct > 50 else ""
        lines.append(f"  {col:<35}: {cnt:>4} ({pct:5.1f}%){flag}")

    lines.append("\n[ DISTRIBUSI rule_group_primary ]")
    dist = df["rule_group_primary"].value_counts(normalize=True)
    for grp, pct in dist.items():
        bar = "#" * int(pct * 40)
        lines.append(f"  {grp:<30}: {pct:5.1%}  {bar}")

    lines.append("\n[ MITRE COVERAGE ]")
    if "has_mitre" in df.columns:
        n_mitre    = df["has_mitre"].sum()
        n_critical = int(df.get("has_critical_mitre", pd.Series([0])).sum())
        lines.append(f"  has_mitre          : {n_mitre} ({n_mitre/len(df)*100:.1f}%)")
        lines.append(f"  has_critical_mitre : {n_critical} ({n_critical/len(df)*100:.1f}%)")
    else:
        lines.append("  ! WARN: kolom has_mitre tidak ditemukan — periksa parse_jsonl()")

    lines.append("\n[ SRCIP ANALYSIS ]")
    if "srcip_type" in df.columns:
        for t, cnt in df["srcip_type"].value_counts().items():
            lines.append(f"  {t:<10}: {cnt}")
    lines.append("  CATATAN: srcip bukan proxy 'attacker' — lihat konteks srcip_type.")

    lines.append("\n[ BIAS ANALYSIS ]")
    # format='ISO8601' handles mixed formats (with/without microseconds, with/without tz)
    try:
        ts_parsed = pd.to_datetime(df["timestamp_utc"], format="ISO8601", errors="coerce")
        dates = ts_parsed.dt.date.value_counts()
    except Exception:
        # Fallback: parse as string
        dates = df["timestamp_utc"].astype(str).str[:10].value_counts()
    lines.append(f"  Temporal: data dari {len(dates)} hari unik")
    if len(dates) == 1:
        lines.append("  ! TEMPORAL BIAS: satu hari saja — model rentan overfit")
    for agent, pct in df["agent_name"].value_counts(normalize=True).items():
        flag = " ! dominan" if pct > 0.30 else ""
        lines.append(f"  agent {agent:<15}: {pct:5.1%}{flag}")
    if dist.iloc[0] > 0.50:
        lines.append(
            f"  ! RULE IMBALANCE: '{dist.index[0]}' = {dist.iloc[0]:.1%}"
            " — IF bisa jadi rule-group detector"
        )

    lines.append("\n[ AGENT CRITICALITY ]")
    lines.append("  DOMAIN ASSUMPTION: tabel ini spesifik UPT TIK INSTIKI")
    lines.append("  Dokumentasikan sebagai 'Tabel Pemetaan Aset Kritis' di Bab 3")
    if "criticality_score" in df.columns:
        for s, cnt in df["criticality_score"].value_counts().sort_index().items():
            lines.append(f"  score={s}: {cnt}")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(base_dir: Path = BASE_DIR) -> None:
    log.info("=== JSON → PIPELINE (BATCH PER BULAN) ===")

    all_csv_files: list[Path] = []

    # Struktur: base_dir → year_dir → month_dir → hasil_json → *.json
    for year_dir in sorted(base_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue

            hasil_dir  = month_dir / "hasil_json"
            json_files = list(hasil_dir.rglob("*.json")) if hasil_dir.exists() else []

            if not json_files:
                log.debug("Skip %s — tidak ada file JSON", month_dir)
                continue

            log.info("[PROSES BULAN] %s/%s — %d file", year_dir.name, month_dir.name, len(json_files))

            records_full_all: list[dict] = []
            records_min_all:  list[dict] = []

            for file in json_files:
                log.info("  parsing %s", file.name)
                records_full, records_min, stats = parse_jsonl(file)
                stats.report()
                records_full_all.extend(records_full)
                records_min_all.extend(records_min)

            if not records_full_all:
                log.warning("Tidak ada record valid dari %s", month_dir.name)
                continue

            df_full = build_df(records_full_all, DEDUP_KEY)
            df_min  = build_df(records_min_all,  DEDUP_KEY)

            tahun = year_dir.name
            bulan = month_dir.name

            out_full = month_dir / f"rbta_ready_{tahun}_{bulan}.csv"
            out_min  = month_dir / f"rbta_raw_{tahun}_{bulan}.csv"

            df_full.to_csv(out_full, index=False)
            df_min.to_csv(out_min, index=False)
            log.info("[DONE] %s (%d rows)", out_full.name, len(df_full))
            all_csv_files.append(out_full)

    if not all_csv_files:
        log.error("Tidak ada CSV yang dihasilkan. Pipeline dihentikan.")
        return

    log.info("=== MERGING %d CSV ===", len(all_csv_files))
    df_all = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in all_csv_files], ignore_index=True
    )

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=[DEDUP_KEY])
    log.info("[GLOBAL DEDUP] %d duplikat dihapus", before - len(df_all))

    df_all = df_all.sort_values("timestamp_utc").reset_index(drop=True)

    final_out = base_dir / "rbta_ready_ALL.csv"
    df_all.to_csv(final_out, index=False)
    log.info("[FINAL] %s (%d rows)", final_out, len(df_all))

    report = data_quality_report(df_all)
    log.info("\n%s", report)
    report_path = base_dir / "data_quality_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("[REPORT] disimpan: %s", report_path)


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR
    main(base_dir=root)