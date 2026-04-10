"""
isolation_forest.py  —  13-fitur + Decision Matrix + False Positive Gate
=========================================================================
Perubahan dari versi sebelumnya:

  [NEW-1] FEATURE_COLS diperluas dari 9 menjadi 13 fitur
      f10  rule_group_entropy       -- behavioral diversity (dari feature_engineering)
      f11  tactic_progression_score -- kill chain ordering (dari feature_engineering)
      f12  deviation_from_baseline  -- deviasi dari rolling average (dari feature_engineering)
      f13  cross_agent_spread       -- lateral movement proxy (dari feature_engineering)

  [NEW-2] run_pipeline() memanggil enrich_features() dari feature_engineering.py
      setelah add_if_features(). Urutan: load → f1-f9 → f10-f13 → IF → decision.

  [NEW-3] Visualisasi panel F diperbarui untuk 13 fitur (bar chart lebih sempit).

  [COMPAT] Decision Matrix, False Positive Gate, SOAR payload tidak berubah.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
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

# ── 13 Fitur Isolation Forest ─────────────────────────────────────────────────
FEATURE_COLS = [
    "alert_count",               # f1  volume dalam bucket
    "max_severity",              # f2  rule.level tertinggi
    "duration_sec",              # f3  durasi insiden
    "attacker_count",            # f4  IP penyerang unik
    "rule_group_severity_enc",   # f5  ordinal encoding rule_group
    "agent_criticality",         # f6  bobot kritis aset (1-4)
    "hour_of_day",               # f7  jam kejadian
    "unique_rules_triggered",    # f8  keberagaman rule_id
    "mitre_hit_count",           # f9  jumlah alert bersinyal MITRE
    "rule_group_entropy",        # f10 Shannon entropy distribusi rule_group
    "tactic_progression_score",  # f11 urutan taktik MITRE di kill chain
    "deviation_from_baseline",   # f12 deviasi dari rolling avg 24 jam
    "cross_agent_spread",        # f13 jumlah agent dengan srcip yang sama
]

FEATURE_LABELS = {
    "alert_count":               "f1 · Alert count",
    "max_severity":              "f2 · Max severity",
    "duration_sec":              "f3 · Duration (s)",
    "attacker_count":            "f4 · Attacker count",
    "rule_group_severity_enc":   "f5 · Rule group enc",
    "agent_criticality":         "f6 · Agent criticality",
    "hour_of_day":               "f7 · Hour of day",
    "unique_rules_triggered":    "f8 · Unique rules",
    "mitre_hit_count":           "f9 · MITRE hits",
    "rule_group_entropy":        "f10 · Rule entropy",
    "tactic_progression_score":  "f11 · Tactic progression",
    "deviation_from_baseline":   "f12 · Baseline deviation",
    "cross_agent_spread":        "f13 · Cross-agent spread",
}

# Decision Matrix thresholds
SEVERITY_HIGH_THRESHOLD = 7

DECISION_CRITICAL   = "CRITICAL"
DECISION_SUSPICIOUS = "SUSPICIOUS"
DECISION_NOISE_HIGH = "NOISE_HIGH"
DECISION_NOISE      = "NOISE"
DECISION_CONTEXTUAL = "CONTEXTUAL_ANOMALY"

ACTION_ESCALATE = "ESCALATE"
ACTION_DIGEST   = "DAILY_DIGEST"
ACTION_SUPPRESS = "SUPPRESS"

CLR_NORMAL   = "#4A7FC1"
CLR_ESCALATE = "#D95F3B"
CLR_GRID     = "#E8E8E8"
BG_COLOR     = "#FAFAFA"
ACCENT       = "#2C4F7C"


# ══════════════════════════════════════════════════════════════════════════════
# Load & Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def load_alerts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    return df


def add_if_features(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Validasi dan normalisasi f1-f9. f10-f13 ditambahkan oleh enrich_features()."""
    df = df_meta.copy()
    df["duration_sec"] = df["duration_sec"].clip(lower=0)

    if "rule_group_severity_enc" not in df.columns:
        df["rule_groups"] = df["rule_groups"].astype(str).str.strip().str.lower()
        df["rule_group_severity_enc"] = (
            df["rule_groups"].map(RULE_GROUP_SEVERITY_ENC).fillna(DEFAULT_GROUP_ENC).astype(int)
        )

    if "agent_criticality" not in df.columns:
        df["agent_criticality"] = (
            df["agent_name"].astype(str).str.strip().str.lower()
            .map(AGENT_CRITICALITY).fillna(DEFAULT_CRITICALITY).astype(int)
        )
    else:
        df["agent_criticality"] = (
            pd.to_numeric(df["agent_criticality"], errors="coerce")
            .fillna(DEFAULT_CRITICALITY).clip(1, 4).astype(int)
        )

    if "hour_of_day" not in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["hour_of_day"] = df["start_time"].dt.hour

    for col in ("unique_rules_triggered", "mitre_hit_count"):
        if col not in df.columns:
            log.warning("Kolom %s tidak ada — di-set 0.", col)
            df[col] = 0

    # f1-f9 harus integer
    for col in FEATURE_COLS[:9]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["rule_groups"] = df["rule_groups"].astype(str).str.strip().str.lower()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Training & Scoring
# ══════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(
    df:            pd.DataFrame,
    contamination: float = 0.05,
    n_estimators:  int   = 200,
    random_state:  int   = 42,
) -> tuple[IsolationForest, RobustScaler, np.ndarray, np.ndarray]:
    """
    Latih Isolation Forest pada feature matrix [f1..f13].
    RobustScaler dipilih karena distribusi fitur SIEM sangat skewed.
    f12 (deviation_from_baseline) bisa negatif — RobustScaler aman untuk ini.
    """
    # Pastikan semua kolom ada — fallback ke 0 jika tidak
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Fitur tidak ditemukan (di-set 0): %s", missing)
        for c in missing:
            df[c] = 0

    X        = df[FEATURE_COLS].values.astype(float)
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
    inv  = -raw_scores
    vmin, vmax = inv.min(), inv.max()
    if vmax - vmin < 1e-10:
        return np.full_like(inv, 0.5)
    return ((inv - vmin) / (vmax - vmin)).clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Threshold
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(scores: np.ndarray, method: str = "iqr") -> float:
    if method == "iqr":
        q1, q3 = np.percentile(scores, [25, 75])
        theta   = min(q3 + 1.5 * (q3 - q1), 1.0)
    elif method == "percentile":
        theta = float(np.percentile(scores, 95))
    else:
        raise ValueError(f"method harus 'iqr' atau 'percentile', bukan '{method}'")
    return round(float(theta), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Decision Matrix + False Positive Gate
# ══════════════════════════════════════════════════════════════════════════════

def _false_positive_gate(row: pd.Series, score: float, theta: float) -> bool:
    """
    True jika baris harus direklasifikasi ke CONTEXTUAL_ANOMALY.

    Kondisi: score tinggi TAPI tidak ada konteks serangan:
      - Tidak ada IP eksternal (attacker_count == 0)
      - Tidak ada sinyal MITRE (mitre_hit_count == 0)
      - Tidak ada cross-agent spread (cross_agent_spread <= 1)
      - Severity rendah (max_severity < SEVERITY_HIGH_THRESHOLD)

    Ini mencegah syscheck/rootcheck rutin dieskalasi hanya karena
    entropy tinggi atau volume yang tidak biasa.
    """
    if score < theta:
        return False
    return (
        int(row.get("attacker_count", 0))      == 0 and
        int(row.get("mitre_hit_count", 0))     == 0 and
        int(row.get("cross_agent_spread", 0))  <= 1 and
        int(row.get("max_severity", 0))        < SEVERITY_HIGH_THRESHOLD
    )


def apply_decision_matrix(
    df:     pd.DataFrame,
    scores: np.ndarray,
    theta:  float,
) -> pd.DataFrame:
    """
    Decision Matrix 4 kuadran + False Positive Gate.

    Q1: score >= theta  AND  max_severity >= HIGH  -> CRITICAL   -> ESCALATE
    Q2: score >= theta  AND  max_severity <  HIGH  -> SUSPICIOUS -> ESCALATE
    Q3: score <  theta  AND  max_severity >= HIGH  -> NOISE_HIGH -> DAILY_DIGEST
    Q4: score <  theta  AND  max_severity <  HIGH  -> NOISE      -> SUPPRESS

    FP Gate (override Q1/Q2): no external IP + no MITRE + no spread + low sev
                               -> CONTEXTUAL_ANOMALY -> SUPPRESS
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
# SOAR Payload
# ══════════════════════════════════════════════════════════════════════════════

def build_soar_payload(row: pd.Series) -> dict:
    """JSON webhook payload standar untuk SOAR Shuffle."""
    return {
        "event_id":              str(row.get("meta_id", "")),
        "anomaly_score":         float(row.get("anomaly_score", 0.0)),
        "severity":              str(row.get("decision", "")),
        "agent":                 str(row.get("agent_name", "")),
        "rule_groups":           str(row.get("rule_groups", "")),
        "alert_count":           int(row.get("alert_count", 0)),
        "max_severity":          int(row.get("max_severity", 0)),
        "mitre_hits":            int(row.get("mitre_hit_count", 0)),
        "rule_entropy":          float(row.get("rule_group_entropy", 0.0)),
        "tactic_progression":    float(row.get("tactic_progression_score", 0.0)),
        "baseline_deviation":    float(row.get("deviation_from_baseline", 0.0)),
        "cross_agent_spread":    int(row.get("cross_agent_spread", 0)),
        "decision":              str(row.get("action", "")),
        "reason": (
            f"score={row.get('anomaly_score', 0):.4f} "
            f"sev={row.get('max_severity', 0)} "
            f"mitre={row.get('mitre_hit_count', 0)} "
            f"entropy={row.get('rule_group_entropy', 0):.3f} "
            f"spread={row.get('cross_agent_spread', 0)} "
            f"dev={row.get('deviation_from_baseline', 0):.3f}"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, theta: float) -> str:
    n_total    = len(df)
    n_escalate = (df["action"] == ACTION_ESCALATE).sum()
    n_digest   = (df["action"] == ACTION_DIGEST).sum()
    n_suppress = (df["action"] == ACTION_SUPPRESS).sum()
    n_ctx      = (df["decision"] == DECISION_CONTEXTUAL).sum()

    top10 = (
        df[df["escalate"] == 1][
            ["meta_id", "agent_name", "rule_groups", "alert_count",
             "max_severity", "mitre_hit_count", "rule_group_entropy",
             "deviation_from_baseline", "cross_agent_spread",
             "anomaly_score", "decision"]
        ]
        .sort_values("anomaly_score", ascending=False)
        .head(10)
    )

    lines = [
        "",
        "=" * 70,
        "  LAPORAN ISOLATION FOREST — 13 FITUR + DECISION MATRIX",
        "=" * 70,
        f"  Total Meta-Alert           : {n_total:,}",
        f"  Threshold theta            : {theta}",
        f"  CRITICAL / SUSPICIOUS      : {n_escalate:,}  ({n_escalate/n_total*100:.1f}%)",
        f"  NOISE_HIGH (daily digest)  : {n_digest:,}  ({n_digest/n_total*100:.1f}%)",
        f"  NOISE (suppress)           : {n_suppress:,}  ({n_suppress/n_total*100:.1f}%)",
        f"  CONTEXTUAL_ANOMALY         : {n_ctx:,}  ({n_ctx/n_total*100:.1f}%)",
        f"  Efek reduksi notifikasi    : {(n_suppress+n_ctx)/n_total*100:.2f}%",
        "",
        "  Distribusi score:",
        f"    Min    : {df['anomaly_score'].min():.4f}",
        f"    Median : {df['anomaly_score'].median():.4f}",
        f"    Q3     : {df['anomaly_score'].quantile(0.75):.4f}",
        f"    Max    : {df['anomaly_score'].max():.4f}",
        "",
        "  Top 10 Meta-Alert ESCALATE:",
        "-" * 70,
    ]
    for _, row in top10.iterrows():
        lines.append(
            f"  id={int(row['meta_id']):>5} | {row['agent_name']:<14} | "
            f"{row['rule_groups']:<22} | cnt={int(row['alert_count']):>4} | "
            f"sev={int(row['max_severity']):>2} | mitre={int(row['mitre_hit_count']):>3} | "
            f"ent={float(row['rule_group_entropy']):.2f} | "
            f"dev={float(row['deviation_from_baseline']):+.2f} | "
            f"spr={int(row['cross_agent_spread']):>2} | "
            f"score={row['anomaly_score']:.4f} | {row['decision']}"
        )

    lines += ["-" * 70, "", "  Distribusi decision:"]
    for dec, cnt in df["decision"].value_counts().items():
        lines.append(f"    {dec:<25}: {cnt:>4}")

    lines.append("=" * 70)
    report = "\n".join(lines)
    log.info(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi (6 panel)
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title: str) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=10, fontweight="bold",
                 color=ACCENT, pad=8, loc="left")
    ax.tick_params(colors="#555", labelsize=7)
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
    """6-panel visualization untuk 13 fitur."""
    fig = plt.figure(figsize=(20, 14), facecolor="white")
    fig.suptitle(
        "Isolation Forest — 13 Fitur Behavioral + Decision Matrix · INSTIKI SOC",
        fontsize=13, fontweight="bold", color=ACCENT, y=0.98,
    )
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.50, wspace=0.40,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
    )

    scores   = df["anomaly_score"].values
    escalate = df["escalate"].values

    # [A] Distribusi anomaly_score
    ax_a = fig.add_subplot(gs[0, :2])
    _style_ax(ax_a, "A · Distribusi anomaly_score (Decision Matrix 4 Kuadran)")
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

    # [B] Scatter alert_count vs rule_group_entropy (f1 vs f10)
    ax_b = fig.add_subplot(gs[0, 2])
    _style_ax(ax_b, "B · Alert count vs Rule entropy (f10)")
    ent_col = "rule_group_entropy" if "rule_group_entropy" in df.columns else "max_severity"
    sc = ax_b.scatter(
        df["alert_count"], df[ent_col],
        c=scores, cmap="RdYlBu_r", s=16, alpha=0.65,
        linewidths=0, vmin=0, vmax=1, zorder=2,
    )
    cb = plt.colorbar(sc, ax=ax_b, shrink=0.85, pad=0.02)
    cb.set_label("anomaly_score", fontsize=7, color="#555")
    cb.ax.tick_params(labelsize=7)
    ax_b.set_xlabel("alert_count", fontsize=9, color="#555")
    ax_b.set_ylabel(ent_col, fontsize=9, color="#555")

    # [C] Top rule_groups yang dieskalasi
    ax_c = fig.add_subplot(gs[1, 0])
    _style_ax(ax_c, "C · Eskalasi per rule_groups")
    rg_esc = df[df["escalate"] == 1]["rule_groups"].value_counts().head(10)
    bars_c = ax_c.barh(rg_esc.index[::-1], rg_esc.values[::-1],
                       color=CLR_ESCALATE, alpha=0.85, height=0.65, zorder=2)
    ax_c.set_xlabel("Jumlah eskalasi", fontsize=8, color="#555")
    for bar, val in zip(bars_c, rg_esc.values[::-1]):
        ax_c.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                  str(val), va="center", fontsize=7, color="#555")
    ax_c.grid(axis="x", color=CLR_GRID, linewidth=0.8)
    ax_c.grid(axis="y", visible=False)

    # [D] Decision distribution
    ax_d = fig.add_subplot(gs[1, 1])
    _style_ax(ax_d, "D · Distribusi Decision")
    dec_counts = df["decision"].value_counts()
    colors_d   = {
        DECISION_CRITICAL:   "#C0392B",
        DECISION_SUSPICIOUS: "#E67E22",
        DECISION_NOISE_HIGH: "#F1C40F",
        DECISION_CONTEXTUAL: "#95A5A6",
        DECISION_NOISE:      CLR_NORMAL,
    }
    bars_d = ax_d.bar(
        range(len(dec_counts)),
        dec_counts.values,
        color=[colors_d.get(k, CLR_NORMAL) for k in dec_counts.index],
        alpha=0.85, zorder=2,
    )
    ax_d.set_xticks(range(len(dec_counts)))
    ax_d.set_xticklabels(
        [k.replace("_", "\n") for k in dec_counts.index],
        fontsize=6, rotation=0,
    )
    ax_d.set_ylabel("Jumlah meta-alert", fontsize=8, color="#555")
    for bar, val in zip(bars_d, dec_counts.values):
        ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  str(val), ha="center", va="bottom", fontsize=7)

    # [E] PCA 2D scatter
    ax_e = fig.add_subplot(gs[1, 2])
    _style_ax(ax_e, "E · PCA 2D — normal vs anomali (13 fitur)")
    pca     = PCA(n_components=2, random_state=42)
    X_pca   = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    ax_e.scatter(X_pca[escalate == 0, 0], X_pca[escalate == 0, 1],
                 c=CLR_NORMAL, s=10, alpha=0.35, label="Normal", linewidths=0, zorder=2)
    ax_e.scatter(X_pca[escalate == 1, 0], X_pca[escalate == 1, 1],
                 c=CLR_ESCALATE, s=20, alpha=0.85, label="Anomali", linewidths=0, zorder=3)
    ax_e.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=8, color="#555")
    ax_e.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=8, color="#555")
    ax_e.legend(fontsize=7, framealpha=0.9, markerscale=1.5)

    # [F] Feature importance — 13 fitur (bar lebih sempit)
    ax_f = fig.add_subplot(gs[2, :])
    _style_ax(ax_f, "F · Kontribusi fitur — rata-rata nilai Anomali vs Normal (13 fitur)")
    avail_feats = [c for c in FEATURE_COLS if c in df.columns]
    df_feat     = df[avail_feats].copy().astype(float)
    for col in avail_feats:
        col_max = df_feat[col].abs().max()
        if col_max > 0:
            df_feat[col] = df_feat[col] / col_max

    mean_esc = df_feat[escalate == 1].mean()
    mean_nor = df_feat[escalate == 0].mean()
    x      = np.arange(len(avail_feats))
    w      = 0.30
    labels = [FEATURE_LABELS.get(c, c) for c in avail_feats]

    bars_e1 = ax_f.bar(x - w / 2, mean_esc.values, w, color=CLR_ESCALATE,
                       alpha=0.85, label="Anomali (escalate)", zorder=2)
    bars_e0 = ax_f.bar(x + w / 2, mean_nor.values, w, color=CLR_NORMAL,
                       alpha=0.75, label="Normal (suppress)", zorder=2)
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    ax_f.set_ylabel("Rata-rata nilai ternormalisasi", fontsize=9, color="#555")
    ax_f.legend(fontsize=8, framealpha=0.9)

    for bar in bars_e1:
        h = bar.get_height()
        if abs(h) > 0.02:
            ax_f.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                      f"{h:.2f}", ha="center", va="bottom", fontsize=6, color=CLR_ESCALATE)
    for bar in bars_e0:
        h = bar.get_height()
        if abs(h) > 0.02:
            ax_f.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                      f"{h:.2f}", ha="center", va="bottom", fontsize=6, color=CLR_NORMAL)

    n_esc = int(escalate.sum())
    n_tot = len(escalate)
    fig.text(
        0.5, 0.005,
        f"Total: {n_tot}  |  Eskalasi: {n_esc} ({n_esc/n_tot*100:.1f}%)  |  "
        f"theta={theta}  |  13 fitur (f1-f13)  |  Decision Matrix 4 Kuadran + FP Gate",
        ha="center", fontsize=7.5, color="#888",
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    log.info("Visualisasi disimpan: %s", output_path)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    csv_path:       str        = "output/meta_alerts_rbta.csv",
    output_dir:     str        = "output/",
    contamination:  float      = 0.05,
    n_estimators:   int        = 200,
    theta_method:   str        = "iqr",
    theta_override: float | None = None,
    random_state:   int        = 42,
) -> tuple[pd.DataFrame, IsolationForest, RobustScaler, float]:
    os.makedirs(output_dir, exist_ok=True)
    log.info("=== ISOLATION FOREST — 13 fitur + Decision Matrix ===")
    log.info("Input: %s", csv_path)

    # Step 1: Load
    df = load_alerts(csv_path)

    # Step 2: f1-f9
    df = add_if_features(df)
    log.info("%d meta-alert dimuat. Menambahkan f10-f13 ...", len(df))

    # Step 3: f10-f13 (behavioral)
    try:
        from engine.feature_engineering import enrich_features
        df = enrich_features(df)
    except ImportError:
        log.warning(
            "feature_engineering.py tidak ditemukan di src/. "
            "Mencoba import langsung ..."
        )
        try:
            from engine.feature_engineering import enrich_features
            df = enrich_features(df)
        except ImportError:
            log.error(
                "Tidak bisa import feature_engineering. "
                "f10-f13 di-set 0. Pastikan file ada di src/."
            )
            for col in ("rule_group_entropy", "tactic_progression_score",
                        "deviation_from_baseline", "cross_agent_spread"):
                df[col] = 0

    # Pastikan semua fitur ada dan bertipe numerik
    for col in FEATURE_COLS:
        if col not in df.columns:
            log.warning("Fitur %s tidak ada setelah enrich — di-set 0.", col)
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    log.info("Feature matrix 13-kolom:\n%s",
             df[FEATURE_COLS].describe().round(3).to_string())

    # Step 4: Training
    log.info(
        "Training IF (n_estimators=%d, contamination=%.2f) ...",
        n_estimators, contamination,
    )
    model, scaler, X_scaled, raw_scores = train_isolation_forest(
        df,
        contamination = contamination,
        n_estimators  = n_estimators,
        random_state  = random_state,
    )
    scores = normalize_scores(raw_scores)
    log.info("Score range: [%.4f, %.4f]", scores.min(), scores.max())

    # Step 5: Threshold
    if theta_override is not None:
        theta = float(theta_override)
        log.info("Threshold theta = %.4f (manual override)", theta)
    else:
        theta = find_optimal_threshold(scores, method=theta_method)
        log.info("Threshold theta = %.4f (metode: %s)", theta, theta_method)

    # Step 6: Decision Matrix
    df_scored = apply_decision_matrix(df, scores, theta)

    n_esc = df_scored["escalate"].sum()
    n_sup = len(df_scored) - n_esc
    log.info(
        "Eskalasi: %d (%.1f%%)  |  Lainnya: %d (%.1f%%)",
        n_esc, n_esc / len(df_scored) * 100,
        n_sup, n_sup / len(df_scored) * 100,
    )

    # Step 7: Report
    report = print_report(df_scored, theta)
    with open(os.path.join(output_dir, "anomaly_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Step 8: SOAR payloads
    soar_payloads = [
        build_soar_payload(row)
        for _, row in df_scored[df_scored["escalate"] == 1].iterrows()
    ]
    soar_path = os.path.join(output_dir, "soar_payloads.json")
    with open(soar_path, "w", encoding="utf-8") as f:
        json.dump(soar_payloads, f, indent=2, default=str)
    log.info("SOAR payloads: %s (%d entri)", soar_path, len(soar_payloads))

    # Step 9: CSV scored
    export_base = [
        "meta_id", "agent_id", "agent_name", "rule_groups",
        "start_time", "end_time", "duration_sec",
        "alert_count", "max_severity", "attacker_count",
        "external_threat_count", "internal_src_count",
    ]
    export_cols = (
        [c for c in export_base if c in df_scored.columns] +
        [c for c in FEATURE_COLS if c in df_scored.columns] +
        ["anomaly_score", "decision", "action", "escalate"]
    )
    df_scored[export_cols].to_csv(
        os.path.join(output_dir, "meta_alerts_scored.csv"), index=False
    )

    # Step 10: Visualisasi
    log.info("Membuat visualisasi 6-panel ...")
    visualize(df_scored, X_scaled, theta,
              output_path=os.path.join(output_dir, "visualisasi_if.png"))

    return df_scored, model, scaler, theta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    CSV_INPUT  = sys.argv[1] if len(sys.argv) > 1 else "output/meta_alerts_rbta.csv"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "output/"
    df_scored, model, scaler, theta = run_pipeline(
        csv_path=CSV_INPUT, output_dir=OUTPUT_DIR,
    )
    log.info("Pipeline selesai.")