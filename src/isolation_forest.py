"""
isolation_forest.py  —  9-fitur + Decision Matrix dengan False Positive Gate
=============================================================================
Perubahan dari versi sebelumnya:

  [NEW-1] FEATURE_COLS diperluas dari 7 menjadi 9 fitur
      Ditambahkan:
        f8  unique_rules_triggered  -- keberagaman rules (proxy APT)
        f9  mitre_hit_count         -- sinyal taktik MITRE ATT&CK

  [NEW-2] Decision Matrix kuadran menggantikan threshold linier
      Empat kuadran berbasis (anomaly_score, max_severity):
        Q1 HIGH-score + HIGH-sev  -> CRITICAL   -> ESCALATE segera
        Q2 HIGH-score + LOW-sev   -> SUSPICIOUS -> ESCALATE (APT candidate)
        Q3 LOW-score  + HIGH-sev  -> NOISE_HIGH -> Daily Digest (suppress notif)
        Q4 LOW-score  + LOW-sev   -> NOISE      -> Suppress penuh

  [NEW-3] False Positive Gate (Section 7 System Instruction)
      Sebelum ESCALATE: cek apakah anomaly tinggi tapi konteks tidak mendukung.
      Jika semua kondisi terpenuhi (no external IP + no MITRE + low severity)
      maka reklasifikasi ke CONTEXTUAL_ANOMALY dan jangan escalate ke SOAR.

  [NEW-4] build_soar_payload() — output JSON webhook standar SOAR
      Format konsisten untuk Shuffle / webhook endpoint.

  [ADAPT-1..4] — sinkronisasi AGENT_CRITICALITY dan RULE_GROUP_SEVERITY_ENC
      tetap identik dengan rbta_algorithm_02.py.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Konstanta — sinkron dengan rbta_algorithm_02.py
# ══════════════════════════════════════════════════════════════════════════════

AGENT_CRITICALITY: dict[str, int] = {
    "soc-1":         1,
    "pusatkarir":    3,
    "dfir-iris":     4,
    "siput":         2,
    "proxy-manager": 3,
    "e-kuesioner":   2,
    "sads":          3,
    "dvwa":          1,
}
DEFAULT_CRITICALITY = 1

RULE_GROUP_SEVERITY_ENC: dict[str, int] = {
    "ossec":                   1,
    "syslog":                  1,
    "authentication_success":  1,
    "stats":                   1,
    "accesslog":               1,
    "dpkg":                    2,
    "config_changed":          2,
    "virus":                   2,
    "sudo":                    2,
    "pam":                     2,
    "sca":                     2,
    "sca_check":               2,
    "rootcheck":               3,
    "syscheck_file":           3,
    "syscheck_entry_deleted":  3,
    "syscheck_entry_added":    3,
    "system_error":            3,
    "docker-error":            3,
    "windows":                 3,
    "virustotal":              3,
    "web":                     4,
    "apache":                  4,
    "nginx":                   4,
    "authentication_failed":   4,
    "attack":                  5,
    "sql_injection":           6,
    "vulnerability-detector":  6,
    "judol_file":              6,
}
DEFAULT_GROUP_ENC = 2

# ── 9 Fitur Isolation Forest ──────────────────────────────────────────────────
FEATURE_COLS = [
    "alert_count",               # f1
    "max_severity",              # f2
    "duration_sec",              # f3
    "attacker_count",            # f4
    "rule_group_severity_enc",   # f5
    "agent_criticality",         # f6
    "hour_of_day",               # f7
    "unique_rules_triggered",    # f8 [BARU]
    "mitre_hit_count",           # f9 [BARU]
]

FEATURE_LABELS = {
    "alert_count":             "f1 · Alert count",
    "max_severity":            "f2 · Max severity",
    "duration_sec":            "f3 · Duration (s)",
    "attacker_count":          "f4 · Attacker count",
    "rule_group_severity_enc": "f5 · Rule group enc",
    "agent_criticality":       "f6 · Agent criticality",
    "hour_of_day":             "f7 · Hour of day",
    "unique_rules_triggered":  "f8 · Unique rules",
    "mitre_hit_count":         "f9 · MITRE hit count",
}

CLR_NORMAL   = "#4A7FC1"
CLR_ESCALATE = "#D95F3B"
CLR_GRID     = "#E8E8E8"
BG_COLOR     = "#FAFAFA"
ACCENT       = "#2C4F7C"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def load_alerts(csv_path: str) -> pd.DataFrame:
    """Load CSV Meta-Alert dan normalisasi timestamp ke UTC."""
    df = pd.read_csv(csv_path)
    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    return df


def add_if_features(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan / validasi 9 fitur numerik untuk Isolation Forest.

    f8 dan f9 diambil langsung dari kolom yang dihasilkan oleh
    run_rbta() v4. Jika tidak ada (backward compatibility), di-set 0
    dengan warning.
    """
    df = df_meta.copy()

    df["duration_sec"] = df["duration_sec"].clip(lower=0)

    # f5
    if "rule_group_severity_enc" not in df.columns:
        df["rule_groups"] = df["rule_groups"].astype(str).str.strip().str.lower()
        df["rule_group_severity_enc"] = (
            df["rule_groups"].map(RULE_GROUP_SEVERITY_ENC)
            .fillna(DEFAULT_GROUP_ENC)
            .astype(int)
        )

    # f6
    if "agent_criticality" not in df.columns:
        df["agent_criticality"] = (
            df["agent_name"].astype(str).str.strip().str.lower()
            .map(AGENT_CRITICALITY)
            .fillna(DEFAULT_CRITICALITY)
            .astype(int)
        )
    else:
        df["agent_criticality"] = (
            pd.to_numeric(df["agent_criticality"], errors="coerce")
            .fillna(DEFAULT_CRITICALITY)
            .clip(1, 4)
            .astype(int)
        )

    # f7
    if "hour_of_day" not in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_localize(None)
        df["hour_of_day"] = df["start_time"].dt.hour

    # f8
    if "unique_rules_triggered" not in df.columns:
        log.warning("Kolom unique_rules_triggered tidak ada — di-set 0.")
        df["unique_rules_triggered"] = 0

    # f9
    if "mitre_hit_count" not in df.columns:
        log.warning("Kolom mitre_hit_count tidak ada — di-set 0.")
        df["mitre_hit_count"] = 0

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["rule_groups"] = df["rule_groups"].astype(str).str.strip().str.lower()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Training & Scoring
# ══════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(
    df:            pd.DataFrame,
    contamination: float = 0.05,
    n_estimators:  int   = 200,
    random_state:  int   = 42,
) -> tuple[IsolationForest, RobustScaler, np.ndarray, np.ndarray]:
    """
    Latih Isolation Forest pada feature matrix [f1..f9].

    RobustScaler dipilih karena distribusi fitur SIEM sangat skewed
    (alert_count, duration_sec: long-tail tinggi).
    """
    X = df[FEATURE_COLS].values
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    model    = IsolationForest(
        n_estimators  = n_estimators,
        contamination = contamination,
        random_state  = random_state,
        n_jobs        = -1,
    )
    model.fit(X_scaled)
    scores = model.score_samples(X_scaled)
    return model, scaler, X_scaled, scores


def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Konversi raw sklearn score [-1, 0] ke anomaly_score [0, 1]."""
    inv  = -raw_scores
    vmin, vmax = inv.min(), inv.max()
    if vmax - vmin < 1e-10:
        return np.full_like(inv, 0.5)
    return ((inv - vmin) / (vmax - vmin)).clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Threshold
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(
    scores: np.ndarray,
    method: str = "iqr",
) -> float:
    if method == "iqr":
        q1, q3 = np.percentile(scores, [25, 75])
        theta   = min(q3 + 1.5 * (q3 - q1), 1.0)
    elif method == "percentile":
        theta = float(np.percentile(scores, 95))
    else:
        raise ValueError(f"method harus 'iqr' atau 'percentile', bukan '{method}'")
    return round(float(theta), 4)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Decision Matrix (Kuadran) + False Positive Gate
# ══════════════════════════════════════════════════════════════════════════════

# Threshold severity: rule_level >= ini dianggap HIGH
SEVERITY_HIGH_THRESHOLD = 7

# Label keputusan
DECISION_CRITICAL     = "CRITICAL"
DECISION_SUSPICIOUS   = "SUSPICIOUS"
DECISION_NOISE_HIGH   = "NOISE_HIGH"
DECISION_NOISE        = "NOISE"
DECISION_CONTEXTUAL   = "CONTEXTUAL_ANOMALY"

ACTION_ESCALATE       = "ESCALATE"
ACTION_DIGEST         = "DAILY_DIGEST"
ACTION_SUPPRESS       = "SUPPRESS"


def _false_positive_gate(row: pd.Series, score: float, theta: float) -> bool:
    """
    Kembalikan True jika baris ini harus direklasifikasi ke CONTEXTUAL_ANOMALY.

    Kondisi: anomaly_score tinggi TAPI:
      - Tidak ada IP eksternal (attacker_count == 0)
      - Tidak ada sinyal MITRE (mitre_hit_count == 0)
      - Severity rendah (max_severity < SEVERITY_HIGH_THRESHOLD)

    Ini mencegah log sistem biasa (syscheck, rootcheck) dieskalasi
    hanya karena volume atau pola waktu yang tidak biasa.
    """
    if score < theta:
        return False
    no_external = int(row.get("attacker_count", 0)) == 0
    no_mitre    = int(row.get("mitre_hit_count", 0)) == 0
    low_sev     = int(row.get("max_severity", 0)) < SEVERITY_HIGH_THRESHOLD
    return no_external and no_mitre and low_sev


def apply_decision_matrix(
    df:     pd.DataFrame,
    scores: np.ndarray,
    theta:  float,
) -> pd.DataFrame:
    """
    Terapkan Decision Matrix kuadran dan False Positive Gate.

    Kuadran:
      Q1  score >= theta  AND  max_severity >= HIGH  -> CRITICAL   -> ESCALATE
      Q2  score >= theta  AND  max_severity <  HIGH  -> SUSPICIOUS -> ESCALATE
      Q3  score <  theta  AND  max_severity >= HIGH  -> NOISE_HIGH -> DAILY_DIGEST
      Q4  score <  theta  AND  max_severity <  HIGH  -> NOISE      -> SUPPRESS

    Setelah Q1/Q2 ditentukan, False Positive Gate berjalan:
      Jika no external IP + no MITRE + low severity
      -> reklasifikasi ke CONTEXTUAL_ANOMALY -> SUPPRESS
    """
    df = df.copy()
    df["anomaly_score"] = np.round(scores, 4)

    decisions: list[str] = []
    actions:   list[str] = []

    for i, row in df.iterrows():
        score    = float(scores[df.index.get_loc(i)])
        high_sev = int(row.get("max_severity", 0)) >= SEVERITY_HIGH_THRESHOLD

        if score >= theta:
            if _false_positive_gate(row, score, theta):
                decisions.append(DECISION_CONTEXTUAL)
                actions.append(ACTION_SUPPRESS)
            elif high_sev:
                decisions.append(DECISION_CRITICAL)
                actions.append(ACTION_ESCALATE)
            else:
                decisions.append(DECISION_SUSPICIOUS)
                actions.append(ACTION_ESCALATE)
        else:
            if high_sev:
                decisions.append(DECISION_NOISE_HIGH)
                actions.append(ACTION_DIGEST)
            else:
                decisions.append(DECISION_NOISE)
                actions.append(ACTION_SUPPRESS)

    df["decision"] = decisions
    df["action"]   = actions
    df["escalate"] = (df["action"] == ACTION_ESCALATE).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SOAR Payload Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_soar_payload(row: pd.Series) -> dict:
    """
    Bangun JSON webhook payload standar untuk SOAR Shuffle.

    Format ini konsisten dengan Section 8 System Instruction.
    Hanya dipanggil untuk baris dengan action == ESCALATE.
    """
    return {
        "event_id":     str(row.get("meta_id", "")),
        "anomaly_score": float(row.get("anomaly_score", 0.0)),
        "severity":      str(row.get("decision", "")),
        "agent":         str(row.get("agent_name", "")),
        "rule_groups":   str(row.get("rule_groups", "")),
        "alert_count":   int(row.get("alert_count", 0)),
        "max_severity":  int(row.get("max_severity", 0)),
        "mitre_hits":    int(row.get("mitre_hit_count", 0)),
        "decision":      str(row.get("action", "")),
        "reason": (
            f"score={row.get('anomaly_score', 0):.4f} "
            f"severity={row.get('max_severity', 0)} "
            f"mitre_hits={row.get('mitre_hit_count', 0)} "
            f"unique_rules={row.get('unique_rules_triggered', 0)}"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, theta: float) -> str:
    n_total    = len(df)
    n_escalate = (df["action"] == ACTION_ESCALATE).sum()
    n_digest   = (df["action"] == ACTION_DIGEST).sum()
    n_suppress = (df["action"] == ACTION_SUPPRESS).sum()
    n_ctx      = (df["decision"] == DECISION_CONTEXTUAL).sum()

    top10 = (
        df[df["escalate"] == 1]
        [["meta_id", "agent_name", "rule_groups",
          "alert_count", "max_severity", "mitre_hit_count",
          "unique_rules_triggered", "anomaly_score", "decision"]]
        .sort_values("anomaly_score", ascending=False)
        .head(10)
    )

    lines = [
        "",
        "=" * 65,
        "  LAPORAN ISOLATION FOREST — RBTA PIPELINE v4",
        "  Decision Matrix: 4 Kuadran + False Positive Gate",
        "=" * 65,
        f"  Total Meta-Alert          : {n_total:,}",
        f"  Threshold theta           : {theta}",
        f"  CRITICAL / SUSPICIOUS     : {n_escalate:,}  ({n_escalate/n_total*100:.1f}%)",
        f"  NOISE_HIGH (daily digest) : {n_digest:,}   ({n_digest/n_total*100:.1f}%)",
        f"  NOISE (suppress)          : {n_suppress:,}  ({n_suppress/n_total*100:.1f}%)",
        f"  CONTEXTUAL_ANOMALY        : {n_ctx:,}   ({n_ctx/n_total*100:.1f}%)",
        f"  Efek reduksi notifikasi   : {(n_suppress+n_ctx)/n_total*100:.2f}%",
        "",
        "  Distribusi score:",
        f"    Min   : {df['anomaly_score'].min():.4f}",
        f"    Median: {df['anomaly_score'].median():.4f}",
        f"    Q3    : {df['anomaly_score'].quantile(0.75):.4f}",
        f"    Max   : {df['anomaly_score'].max():.4f}",
        "",
        "  Top 10 Meta-Alert Anomali (ESCALATE):",
        "-" * 65,
    ]

    for _, row in top10.iterrows():
        lines.append(
            f"  id={int(row['meta_id']):>5} | "
            f"{row['agent_name']:<14} | "
            f"{row['rule_groups']:<22} | "
            f"cnt={int(row['alert_count']):>4} | "
            f"sev={int(row['max_severity']):>2} | "
            f"mitre={int(row['mitre_hit_count']):>3} | "
            f"score={row['anomaly_score']:.4f} | "
            f"{row['decision']}"
        )

    lines += ["-" * 65, "", "  Eskalasi per agent:"]
    for agent, grp in df[df["escalate"] == 1].groupby("agent_name"):
        lines.append(f"    {agent:<16}: {len(grp):>3} insiden")

    lines += ["", "  Distribusi decision:"]
    for dec, cnt in df["decision"].value_counts().items():
        lines.append(f"    {dec:<25}: {cnt:>4}")

    lines.append("=" * 65)

    report = "\n".join(lines)
    log.info(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Visualisasi (6 panel)
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax: object, title: str) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=ACCENT, pad=10, loc="left")
    ax.tick_params(colors="#555", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CLR_GRID)
    ax.spines["bottom"].set_color(CLR_GRID)
    ax.grid(axis="y", color=CLR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def visualize(
    df:          pd.DataFrame,
    X_scaled:    np.ndarray,
    theta:       float,
    output_path: str = "visualisasi_if.png",
) -> None:
    """6-panel visualization."""
    fig = plt.figure(figsize=(18, 13), facecolor="white")
    fig.suptitle(
        "Isolation Forest — 9-Fitur + Decision Matrix · INSTIKI SOC",
        fontsize=14, fontweight="bold", color=ACCENT, y=0.98,
    )
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.38,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
    )

    scores   = df["anomaly_score"].values
    escalate = df["escalate"].values

    # [A] Distribusi anomaly_score
    ax_a = fig.add_subplot(gs[0, :2])
    _style_ax(ax_a, "A · Distribusi anomaly_score (4-kuadran)")
    bins = np.linspace(0, 1, 60)
    ax_a.hist(scores[escalate == 0], bins=bins, color=CLR_NORMAL,
              alpha=0.75, label="Suppress / Digest", zorder=2,
              edgecolor="white", linewidth=0.3)
    ax_a.hist(scores[escalate == 1], bins=bins, color=CLR_ESCALATE,
              alpha=0.85, label="Escalate", zorder=3,
              edgecolor="white", linewidth=0.3)
    ax_a.axvline(theta, color="#333", linestyle="--", linewidth=1.5,
                 label=f"theta = {theta}", zorder=4)
    ax_a.set_xlabel("anomaly_score", fontsize=9, color="#555")
    ax_a.set_ylabel("Frekuensi", fontsize=9, color="#555")
    ax_a.legend(fontsize=8, framealpha=0.9)

    # [B] Scatter alert_count vs max_severity
    ax_b = fig.add_subplot(gs[0, 2])
    _style_ax(ax_b, "B · Alert count vs Severity")
    sc = ax_b.scatter(
        df["alert_count"], df["max_severity"],
        c=scores, cmap="RdYlBu_r", s=18, alpha=0.65,
        linewidths=0, vmin=0, vmax=1, zorder=2,
    )
    cb = plt.colorbar(sc, ax=ax_b, shrink=0.85, pad=0.02)
    cb.set_label("anomaly_score", fontsize=7, color="#555")
    cb.ax.tick_params(labelsize=7)
    ax_b.set_xlabel("alert_count", fontsize=9, color="#555")
    ax_b.set_ylabel("max_severity", fontsize=9, color="#555")

    # [C] Top rule_groups yang dieskalasi
    ax_c = fig.add_subplot(gs[1, 0])
    _style_ax(ax_c, "C · Eskalasi per rule_groups")
    rg_esc = df[df["escalate"] == 1]["rule_groups"].value_counts().head(10)
    bars_c = ax_c.barh(
        rg_esc.index[::-1], rg_esc.values[::-1],
        color=CLR_ESCALATE, alpha=0.85, height=0.65, zorder=2,
    )
    ax_c.set_xlabel("Jumlah eskalasi", fontsize=9, color="#555")
    for bar, val in zip(bars_c, rg_esc.values[::-1]):
        ax_c.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                  str(val), va="center", fontsize=7, color="#555")
    ax_c.grid(axis="x", color=CLR_GRID, linewidth=0.8)
    ax_c.grid(axis="y", visible=False)

    # [D] Eskalasi per agent
    ax_d = fig.add_subplot(gs[1, 1])
    _style_ax(ax_d, "D · Eskalasi per agent (%)")
    agent_esc   = df[df["escalate"] == 1]["agent_name"].value_counts()
    agent_total = df["agent_name"].value_counts()
    agent_pct   = (agent_esc / agent_total * 100).fillna(0).sort_values()
    bar_colors  = [CLR_ESCALATE if v >= 10 else CLR_NORMAL for v in agent_pct.values]
    bars_d = ax_d.barh(
        agent_pct.index, agent_pct.values,
        color=bar_colors, alpha=0.85, height=0.65, zorder=2,
    )
    ax_d.set_xlabel("% dieskalasi", fontsize=9, color="#555")
    for bar, val in zip(bars_d, agent_pct.values):
        ax_d.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}%", va="center", fontsize=7, color="#555")
    ax_d.grid(axis="x", color=CLR_GRID, linewidth=0.8)
    ax_d.grid(axis="y", visible=False)

    # [E] PCA 2D scatter
    ax_e = fig.add_subplot(gs[1, 2])
    _style_ax(ax_e, "E · PCA 2D — normal vs anomali")
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    ax_e.scatter(X_pca[escalate == 0, 0], X_pca[escalate == 0, 1],
                 c=CLR_NORMAL, s=12, alpha=0.4, label="Normal", linewidths=0, zorder=2)
    ax_e.scatter(X_pca[escalate == 1, 0], X_pca[escalate == 1, 1],
                 c=CLR_ESCALATE, s=22, alpha=0.85, label="Anomali", linewidths=0, zorder=3)
    ax_e.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=9, color="#555")
    ax_e.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=9, color="#555")
    ax_e.legend(fontsize=7, framealpha=0.9, markerscale=1.5)

    # [F] Feature importance (9 fitur)
    ax_f = fig.add_subplot(gs[2, :])
    _style_ax(ax_f, "F · Kontribusi fitur — rata-rata nilai pada Anomali vs Normal")
    df_feat = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        col_max = df_feat[col].max()
        if col_max > 0:
            df_feat[col] = df_feat[col] / col_max
    mean_esc = df_feat[escalate == 1].mean()
    mean_nor = df_feat[escalate == 0].mean()
    x      = np.arange(len(FEATURE_COLS))
    w      = 0.35
    labels = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    bars_e1 = ax_f.bar(x - w / 2, mean_esc.values, w, color=CLR_ESCALATE,
                        alpha=0.85, label="Anomali (escalate)", zorder=2)
    bars_e0 = ax_f.bar(x + w / 2, mean_nor.values, w, color=CLR_NORMAL,
                        alpha=0.75, label="Normal (suppress)", zorder=2)
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.5)
    ax_f.set_ylabel("Rata-rata nilai ternormalisasi", fontsize=9, color="#555")
    ax_f.legend(fontsize=9, framealpha=0.9)
    for bar in bars_e1:
        h = bar.get_height()
        ax_f.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                  f"{h:.2f}", ha="center", va="bottom", fontsize=7, color=CLR_ESCALATE)
    for bar in bars_e0:
        h = bar.get_height()
        ax_f.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                  f"{h:.2f}", ha="center", va="bottom", fontsize=7, color=CLR_NORMAL)

    n_esc = int(escalate.sum())
    n_tot = len(escalate)
    fig.text(
        0.5, 0.01,
        f"Total Meta-Alert: {n_tot}  |  Eskalasi: {n_esc} ({n_esc/n_tot*100:.1f}%)  |  "
        f"theta = {theta}  |  9 fitur  |  Decision Matrix: 4 Kuadran + FP Gate",
        ha="center", fontsize=8, color="#888",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    log.info("Visualisasi disimpan: %s", output_path)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    csv_path:       str   = "output/meta_alerts_rbta.csv",
    output_dir:     str   = "output/",
    contamination:  float = 0.05,
    n_estimators:   int   = 200,
    theta_method:   str   = "iqr",
    theta_override: float | None = None,
    random_state:   int   = 42,
) -> tuple[pd.DataFrame, IsolationForest, RobustScaler, float]:
    os.makedirs(output_dir, exist_ok=True)

    log.info("=== ISOLATION FOREST PIPELINE — 9-Fitur + Decision Matrix ===")
    log.info("Input: %s", csv_path)

    df = load_alerts(csv_path)
    df = add_if_features(df)
    log.info("%d Meta-Alert dimuat.", len(df))

    n_neg = (df["duration_sec"] < 0).sum()
    if n_neg:
        log.warning("Masih ada %d duration_sec negatif setelah clip.", n_neg)

    log.info("Feature vector 9-kolom statistik:\n%s",
             df[FEATURE_COLS].describe().round(2).to_string())

    log.info(
        "Training Isolation Forest (n_estimators=%d, contamination=%.2f) ...",
        n_estimators, contamination,
    )
    model, scaler, X_scaled, raw_scores = train_isolation_forest(
        df,
        contamination = contamination,
        n_estimators  = n_estimators,
        random_state  = random_state,
    )
    scores = normalize_scores(raw_scores)
    log.info("Selesai. Score range: [%.4f, %.4f]", scores.min(), scores.max())

    if theta_override is not None:
        theta = float(theta_override)
        log.info("Threshold theta = %.4f (manual override)", theta)
    else:
        theta = find_optimal_threshold(scores, method=theta_method)
        log.info("Threshold theta = %.4f (metode: %s)", theta, theta_method)

    df_scored = apply_decision_matrix(df, scores, theta)

    n_esc = df_scored["escalate"].sum()
    n_sup = len(df_scored) - n_esc
    log.info(
        "Eskalasi: %d (%.1f%%)  |  Suppress/Digest: %d (%.1f%%)",
        n_esc, n_esc/len(df_scored)*100,
        n_sup, n_sup/len(df_scored)*100,
    )

    report = print_report(df_scored, theta)
    report_path = os.path.join(output_dir, "anomaly_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Simpan SOAR payloads untuk semua baris ESCALATE
    soar_payloads = [
        build_soar_payload(row)
        for _, row in df_scored[df_scored["escalate"] == 1].iterrows()
    ]
    soar_path = os.path.join(output_dir, "soar_payloads.json")
    with open(soar_path, "w", encoding="utf-8") as f:
        json.dump(soar_payloads, f, indent=2, default=str)
    log.info("SOAR payloads disimpan: %s (%d entri)", soar_path, len(soar_payloads))

    csv_out     = os.path.join(output_dir, "meta_alerts_scored.csv")
    export_cols = [
        "meta_id", "agent_id", "agent_name", "rule_groups",
        "start_time", "end_time", "duration_sec",
        "alert_count", "max_severity", "attacker_count",
    ] + FEATURE_COLS + [
        "external_threat_count", "internal_src_count",
        "anomaly_score", "decision", "action", "escalate",
    ]
    export_cols = [c for c in export_cols if c in df_scored.columns]
    df_scored[export_cols].to_csv(csv_out, index=False)
    log.info("CSV scored disimpan: %s", csv_out)

    log.info("Membuat visualisasi 6-panel ...")
    viz_path = os.path.join(output_dir, "visualisasi_if.png")
    visualize(df_scored, X_scaled, theta, output_path=viz_path)

    return df_scored, model, scaler, theta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    CSV_INPUT  = sys.argv[1] if len(sys.argv) > 1 else "output/meta_alerts_rbta.csv"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "output/"

    df_scored, model, scaler, theta = run_pipeline(
        csv_path       = CSV_INPUT,
        output_dir     = OUTPUT_DIR,
        contamination  = 0.05,
        n_estimators   = 200,
        theta_method   = "iqr",
        theta_override = None,
        random_state   = 42,
    )
    log.info("Pipeline selesai.")