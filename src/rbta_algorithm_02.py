"""
rbta_algorithm_02.py  —  Enhanced v4  (9-fitur Isolation Forest)
=================================================================
Perubahan dari versi sebelumnya (v3):

  [NEW-1] MetaAlert memiliki 4 field akumulasi baru:
      - unique_rule_ids      : set rule_id dalam bucket (untuk to_dict)
      - external_threat_count: jumlah alert dari IP eksternal
      - internal_src_count   : jumlah alert dari IP internal
      - mitre_hit_count      : jumlah alert yang memiliki has_mitre == 1

  [NEW-2] _process mengakumulasi 4 field baru per event
      Setiap kali sebuah alert dimasukkan ke bucket, field baru
      diperbarui secara inkremental (O(1) per event).

  [NEW-3] to_dict() menghasilkan 9 kolom fitur untuk Isolation Forest
      f1  alert_count
      f2  max_severity
      f3  duration_sec
      f4  attacker_count         (external IPs unik)
      f5  rule_group_severity_enc
      f6  agent_criticality
      f7  hour_of_day
      f8  unique_rules_triggered  [BARU]
      f9  mitre_hit_count         [BARU]

      external_threat_count dan internal_src_count juga diekspor
      sebagai kolom konteks (bukan wajib IF, tapi berguna analisis).

  [ADAPT-1..4] — tidak berubah dari v3 (AGENT_CRITICALITY, RULE_GROUP_SEVERITY_ENC,
      add_if_features, criticality_score pass-through).
"""

import heapq
import logging
import pandas as pd
from dataclasses import dataclass, field
from datetime import timedelta
import time
import json

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Mapping Agent Criticality  —  skala 1-4
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


# ══════════════════════════════════════════════════════════════════════════════
# Ordinal Encoding rule_groups  —  skala 1-6
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# [1]  Out-of-Order Buffer  —  Min-Heap, ukuran k
# ══════════════════════════════════════════════════════════════════════════════

class OutOfOrderBuffer:
    """
    Bounded min-heap buffer berukuran k untuk menangani event out-of-order.

    Kompleksitas:
      - push / pop : O(log k)
      - Per event  : O(log k)
      - Total      : O(n log k),  k << n  (near-linear)
    """

    def __init__(self, k: int = 50) -> None:
        if k < 1:
            raise ValueError("Buffer size k harus >= 1")
        self.k        = k
        self._heap:    list = []
        self._counter: int  = 0

    def push(self, ts: pd.Timestamp, row: object) -> None:
        heapq.heappush(self._heap, (ts, self._counter, row))
        self._counter += 1

    def pop_oldest(self) -> tuple:
        ts, _, row = heapq.heappop(self._heap)
        return ts, row

    def drain_sorted(self) -> list:
        result = []
        while self._heap:
            result.append(self.pop_oldest())
        return result

    @property
    def is_full(self) -> bool:
        return len(self._heap) >= self.k

    def __len__(self) -> int:
        return len(self._heap)


# ══════════════════════════════════════════════════════════════════════════════
# [2]  Elastic Time-Window  —  Adaptive Delta-t berbasis EMA
# ══════════════════════════════════════════════════════════════════════════════

class ElasticWindow:
    """
    Adaptive Delta-t yang dapat melar dan menyusut secara otomatis
    berdasarkan EMA inter-arrival time.

    min_dt dan max_dt relatif terhadap base_dt (default 50% dan 150%).
    Ini mencegah Delta-t 60 menit dipaksa turun ke 1 menit pada dataset besar.
    """

    _WARMUP_SIZE  = 100
    _HIGH_FREQ    = 3.0
    _LOW_FREQ     = 0.33
    _SHRINK_RATE  = 0.80
    _EXPAND_RATE  = 1.20

    def __init__(
        self,
        base_dt:  timedelta,
        min_dt:   timedelta | None = None,
        max_dt:   timedelta | None = None,
        alpha:    float            = 0.10,
    ) -> None:
        self.base_dt    = base_dt
        self.min_dt     = min_dt or timedelta(seconds=base_dt.total_seconds() * 0.5)
        self.max_dt     = max_dt or timedelta(seconds=base_dt.total_seconds() * 1.5)
        self.alpha      = alpha
        self.current_dt = base_dt

        self._last_ts:  pd.Timestamp | None = None
        self._ema:      float | None        = None
        self._baseline: float | None        = None
        self._warmup:   list[float]         = []

        self.dt_history: list[tuple[pd.Timestamp, float]] = []

    def update(self, ts: pd.Timestamp) -> timedelta:
        if self._last_ts is not None:
            gap_sec = (ts - self._last_ts).total_seconds()
            if gap_sec > 0:
                if self._baseline is None:
                    self._warmup.append(gap_sec)
                    if len(self._warmup) >= self._WARMUP_SIZE:
                        self._baseline = sum(self._warmup) / len(self._warmup)
                        self._ema      = self._baseline
                else:
                    self._ema = self.alpha * gap_sec + (1 - self.alpha) * self._ema
                    self._adapt()

        self._last_ts = ts
        self.dt_history.append((ts, self.current_dt.total_seconds() / 60))
        return self.current_dt

    @property
    def current_minutes(self) -> float:
        return self.current_dt.total_seconds() / 60

    @property
    def is_warmed_up(self) -> bool:
        return self._baseline is not None

    def _adapt(self) -> None:
        if not self._ema or self._ema <= 0:
            return
        freq_ratio = self._baseline / self._ema
        if freq_ratio >= self._HIGH_FREQ:
            new_dt      = timedelta(seconds=self.current_dt.total_seconds() * self._SHRINK_RATE)
            self.current_dt = max(new_dt, self.min_dt)
        elif freq_ratio <= self._LOW_FREQ:
            new_dt      = timedelta(seconds=self.current_dt.total_seconds() * self._EXPAND_RATE)
            self.current_dt = min(new_dt, self.max_dt)


# ══════════════════════════════════════════════════════════════════════════════
# [3]  Watermark  —  Event-time watermark untuk late event handling
# ══════════════════════════════════════════════════════════════════════════════

class Watermark:
    """
    Watermark berbasis event-time untuk toleransi clock skew dan late events.
    Model: watermark = max_event_time_seen - max_lateness
    """

    def __init__(self, max_lateness_sec: float = 10.0) -> None:
        if max_lateness_sec < 0:
            raise ValueError("max_lateness_sec harus >= 0")
        self.max_lateness = timedelta(seconds=max_lateness_sec)
        self._max_seen:   pd.Timestamp | None = None
        self.n_on_time:   int = 0
        self.n_late_ok:   int = 0
        self.n_late_drop: int = 0

    def advance(self, ts: pd.Timestamp) -> None:
        if self._max_seen is None or ts > self._max_seen:
            self._max_seen = ts

    def classify(self, ts: pd.Timestamp) -> str:
        wm = self.watermark
        if wm is None or ts >= wm:
            self.n_on_time += 1
            return "on_time"
        hard_cutoff = self._max_seen - 2 * self.max_lateness
        if ts >= hard_cutoff:
            self.n_late_ok += 1
            return "late_ok"
        self.n_late_drop += 1
        return "late_drop"

    @property
    def watermark(self) -> pd.Timestamp | None:
        if self._max_seen is None:
            return None
        return self._max_seen - self.max_lateness


# ══════════════════════════════════════════════════════════════════════════════
# Struktur Data Meta-Alert  —  v4 (9 fitur IF)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetaAlert:
    meta_id:           int
    agent_id:          str
    agent_name:        str
    rule_groups:       str
    start_time:        pd.Timestamp
    end_time:          pd.Timestamp
    parent_meta_id:    int | None = None
    alert_count:       int        = 1
    max_severity:      int        = 0
    criticality_score: int        = 1

    # [NEW-1] Field akumulasi baru
    external_threat_count: int        = 0   # alert dari IP eksternal
    internal_src_count:    int        = 0   # alert dari IP internal
    mitre_hit_count:       int        = 0   # alert yang has_mitre == 1

    srcip_list:    list[str] = field(default_factory=list)
    rule_id_dist:  dict      = field(default_factory=dict)
    severity_dist: dict      = field(default_factory=dict)
    # Set rule_id unik — dikonversi ke count saat to_dict()
    _unique_rule_ids: set = field(default_factory=set, repr=False)

    def to_dict(self) -> dict:
        raw_duration = (self.end_time - self.start_time).total_seconds()
        duration_sec = max(0, int(raw_duration))
        clean_rg     = self.rule_groups.strip().lower()

        agent_crit = self.criticality_score
        if agent_crit == 1:
            agent_crit = AGENT_CRITICALITY.get(
                self.agent_name.strip().lower(), DEFAULT_CRITICALITY
            )

        hour_of_day             = self.start_time.hour
        rule_group_severity_enc = RULE_GROUP_SEVERITY_ENC.get(clean_rg, DEFAULT_GROUP_ENC)

        # f8: jumlah rule_id unik yang terpicu dalam bucket ini
        unique_rules_triggered = len(self._unique_rule_ids)

        return {
            "meta_id":                  self.meta_id,
            "parent_meta_id":           self.parent_meta_id if self.parent_meta_id else "",
            "agent_id":                 self.agent_id,
            "agent_name":               self.agent_name,
            "rule_groups":              clean_rg,
            "start_time":               self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time":                 self.end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_sec":             duration_sec,
            # ── Fitur f1-f9 ──────────────────────────────────────────────
            "alert_count":              self.alert_count,                # f1
            "max_severity":             self.max_severity,               # f2
            # f3: duration_sec (sudah di atas)
            "attacker_count":           len(set(ip for ip in self.srcip_list if ip)),  # f4
            "rule_group_severity_enc":  rule_group_severity_enc,         # f5
            "agent_criticality":        agent_crit,                      # f6
            "hour_of_day":              hour_of_day,                     # f7
            "unique_rules_triggered":   unique_rules_triggered,          # f8 [BARU]
            "mitre_hit_count":          self.mitre_hit_count,            # f9 [BARU]
            # ── Kolom konteks (tidak wajib IF) ────────────────────────────
            "external_threat_count":    self.external_threat_count,
            "internal_src_count":       self.internal_src_count,
            "attacker_ips":             "|".join(sorted(set(ip for ip in self.srcip_list if ip))),
            "severity_dist":            json.dumps(self.severity_dist),
            "rule_id_dist":             json.dumps(self.rule_id_dist),
        }


def _bucket_key(row: pd.Series) -> tuple:
    clean_rg = str(row["rule_groups"]).strip().lower()
    return (str(row["agent_id"]), clean_rg)


def _new_bucket(
    meta_id:   int,
    row:       pd.Series,
    ts:        pd.Timestamp,
    lv:        int,
    ip:        str,
    srcip_type: str,
    parent_id: int | None = None,
) -> MetaAlert:
    rule_id    = str(row.get("rule_id", "unknown"))
    crit_score = int(row.get("criticality_score", 1))
    has_mitre  = int(row.get("has_mitre", 0))

    ext_count = 1 if srcip_type == "external" else 0
    int_count = 1 if srcip_type == "internal" else 0

    ma = MetaAlert(
        meta_id               = meta_id,
        parent_meta_id        = parent_id,
        agent_id              = str(row.get("agent_id", "unknown")),
        agent_name            = str(row.get("agent_name", "")),
        rule_groups           = str(row.get("rule_groups", "unknown")),
        start_time            = ts,
        end_time              = ts,
        alert_count           = 1,
        max_severity          = lv,
        criticality_score     = crit_score,
        external_threat_count = ext_count,
        internal_src_count    = int_count,
        mitre_hit_count       = has_mitre,
        srcip_list            = [ip] if ip else [],
        rule_id_dist          = {rule_id: 1},
        severity_dist         = {str(lv): 1},
    )
    ma._unique_rule_ids = {rule_id}
    return ma


# ══════════════════════════════════════════════════════════════════════════════
# Algoritma RBTA  ——  Enhanced v4
# ══════════════════════════════════════════════════════════════════════════════

def run_rbta(
    df:                 pd.DataFrame,
    delta_t_minutes:    int   = 15,
    max_window_minutes: int   = 60,
    buffer_size:        int   = 50,
    max_lateness_sec:   float = 10.0,
    enable_adaptive:    bool  = True,
) -> tuple[pd.DataFrame, dict[int, list[int]], ElasticWindow, Watermark]:
    """
    RBTA Enhanced v4 dengan 9 fitur Isolation Forest.

    Parameters
    ----------
    df                 : DataFrame dari preprocessing_01 (arrival order).
    delta_t_minutes    : Delta-t awal dalam menit.
    max_window_minutes : Circuit Breaker — durasi maksimal satu bucket.
    buffer_size        : k untuk Out-of-Order Buffer (O(log k)).
    max_lateness_sec   : Toleransi Watermark dalam detik.
    enable_adaptive    : Aktifkan Elastic Time-Window.

    Returns
    -------
    df_meta          : pd.DataFrame  — Meta-alert dengan 9 fitur IF.
    alert_index_map  : dict          — { meta_id: [row_index, ...] }
    elastic          : ElasticWindow — untuk plot adaptive Delta-t.
    wmark            : Watermark     — untuk plot late-event stats.
    """
    base_delta_t   = timedelta(minutes=delta_t_minutes)
    max_window_sec = max_window_minutes * 60

    buffer  = OutOfOrderBuffer(k=buffer_size)
    elastic = ElasticWindow(base_dt=base_delta_t) if enable_adaptive else None
    wmark   = Watermark(max_lateness_sec=max_lateness_sec)

    active_buckets:   dict[tuple, MetaAlert]  = {}
    active_idx_lists: dict[tuple, list[int]]  = {}
    finished_buckets: list[MetaAlert]         = []
    finished_idx:     dict[int, list[int]]    = {}
    meta_id_ctr = 1

    t_start = time.perf_counter()

    def _close_bucket(key: tuple) -> None:
        bucket   = active_buckets.pop(key)
        idx_list = active_idx_lists.pop(key)
        finished_buckets.append(bucket)
        finished_idx[bucket.meta_id] = idx_list

    def _process(ts: pd.Timestamp, row: pd.Series, row_idx: int) -> None:
        nonlocal meta_id_ctr

        wmark.advance(ts)
        status = wmark.classify(ts)
        if status == "late_drop":
            return

        current_dt = elastic.update(ts) if elastic else base_delta_t

        key        = _bucket_key(row)
        lv         = int(row.get("rule_level", 0))
        ip         = str(row.get("srcip", "") or "")
        rule_id    = str(row.get("rule_id", "unknown"))
        srcip_type = str(row.get("srcip_type", "none"))
        has_mitre  = int(row.get("has_mitre", 0))

        if key in active_buckets:
            bucket = active_buckets[key]
            time_since_last = ts - bucket.end_time
            total_duration  = (ts - bucket.start_time).total_seconds()

            fits_window  = time_since_last <= current_dt
            fits_circuit = total_duration  <= max_window_sec

            if fits_window and fits_circuit:
                # Perbarui field standar
                bucket.end_time     = max(bucket.end_time, ts)
                bucket.alert_count += 1
                bucket.max_severity = max(bucket.max_severity, lv)
                if ip:
                    bucket.srcip_list.append(ip)
                bucket.rule_id_dist[rule_id]  = bucket.rule_id_dist.get(rule_id, 0) + 1
                bucket.severity_dist[str(lv)] = bucket.severity_dist.get(str(lv), 0) + 1

                # [NEW-2] Perbarui field akumulasi baru
                bucket._unique_rule_ids.add(rule_id)
                if srcip_type == "external":
                    bucket.external_threat_count += 1
                elif srcip_type == "internal":
                    bucket.internal_src_count += 1
                bucket.mitre_hit_count += has_mitre

                active_idx_lists[key].append(row_idx)
            else:
                is_continued = fits_window and not fits_circuit
                parent_id    = bucket.meta_id if is_continued else None
                _close_bucket(key)
                new_bucket = _new_bucket(
                    meta_id_ctr, row, ts, lv, ip, srcip_type, parent_id
                )
                active_buckets[key]   = new_bucket
                active_idx_lists[key] = [row_idx]
                meta_id_ctr += 1
        else:
            active_buckets[key]   = _new_bucket(
                meta_id_ctr, row, ts, lv, ip, srcip_type
            )
            active_idx_lists[key] = [row_idx]
            meta_id_ctr += 1

    # ── Iterasi utama dengan Out-of-Order Buffer ──────────────────────────────
    for row_idx, row in df.iterrows():
        ts = row["timestamp"]
        buffer.push(ts, (row_idx, row))

        if buffer.is_full:
            old_ts, (old_idx, old_row) = _pop_with_idx(buffer)
            _process(old_ts, old_row, old_idx)

    # ── Drain sisa buffer ─────────────────────────────────────────────────────
    for ts, (old_idx, old_row) in _drain_with_idx(buffer):
        _process(ts, old_row, old_idx)

    # ── Tutup semua bucket yang masih aktif ───────────────────────────────────
    for key in list(active_buckets.keys()):
        _close_bucket(key)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    if not finished_buckets:
        log.warning("RBTA tidak menghasilkan meta-alert. Periksa data input.")
        return pd.DataFrame(), {}, elastic, wmark

    df_meta = (
        pd.DataFrame([m.to_dict() for m in finished_buckets])
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    n_raw  = len(df)
    n_meta = len(df_meta)
    arr    = (1 - n_meta / n_raw) * 100 if n_raw > 0 else 0.0

    log.info(
        "RBTA selesai — raw=%d  meta=%d  ARR=%.2f%%  t=%.2fms",
        n_raw, n_meta, arr, elapsed_ms,
    )

    return df_meta, finished_idx, elastic, wmark


# ══════════════════════════════════════════════════════════════════
# Helper: adaptasi OutOfOrderBuffer untuk payload (idx, row)
# ══════════════════════════════════════════════════════════════════

def _pop_with_idx(buffer: OutOfOrderBuffer) -> tuple:
    ts, _, payload = heapq.heappop(buffer._heap)
    return ts, payload


def _drain_with_idx(buffer: OutOfOrderBuffer) -> list:
    result = []
    while buffer._heap:
        result.append(_pop_with_idx(buffer))
    return result


# ══════════════════════════════════════════════════════════════════
# Validasi mapping
# ══════════════════════════════════════════════════════════════════

def validate_mapping(
    df_raw:          pd.DataFrame,
    df_meta:         pd.DataFrame,
    alert_index_map: dict,
) -> dict:
    all_indices: list[int] = []
    for idx_list in alert_index_map.values():
        all_indices.extend(idx_list)

    total_mapped  = len(all_indices)
    unique_mapped = len(set(all_indices))
    overlap_count = total_mapped - unique_mapped

    map_meta_ids  = set(alert_index_map.keys())
    df_meta_ids   = set(df_meta["meta_id"].tolist()) if "meta_id" in df_meta.columns else set()
    missing_in_df = map_meta_ids - df_meta_ids
    extra_in_map  = df_meta_ids - map_meta_ids

    mismatch_counts: list[dict] = []
    if "meta_id" in df_meta.columns and "alert_count" in df_meta.columns:
        for _, row in df_meta.iterrows():
            mid = row["meta_id"]
            if mid in alert_index_map:
                expected = len(alert_index_map[mid])
                actual   = row["alert_count"]
                if expected != actual:
                    mismatch_counts.append(
                        {"meta_id": mid, "map_len": expected, "df_count": actual}
                    )

    ok = (
        overlap_count      == 0 and
        len(missing_in_df) == 0 and
        len(extra_in_map)  == 0 and
        len(mismatch_counts) == 0
    )

    status = "LULUS" if ok else "GAGAL"
    log.info(
        "Validasi mapping: total_mapped=%d  overlap=%d  missing=%d  extra=%d  "
        "mismatch=%d  status=%s",
        total_mapped, overlap_count, len(missing_in_df),
        len(extra_in_map), len(mismatch_counts), status,
    )

    return {
        "ok":              ok,
        "total_mapped":    total_mapped,
        "total_raw":       len(df_raw),
        "unique_mapped":   unique_mapped,
        "overlap_count":   overlap_count,
        "missing_in_df":   list(missing_in_df),
        "extra_in_map":    list(extra_in_map),
        "mismatch_counts": mismatch_counts,
    }


# ══════════════════════════════════════════════════════════════════
# load_alerts
# ══════════════════════════════════════════════════════════════════

def load_alerts(csv_path: str) -> pd.DataFrame:
    """Load Meta-Alert CSV dan normalisasi timestamp ke UTC."""
    df = pd.read_csv(csv_path)
    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = (
                pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
            )
    if "duration_sec" in df.columns:
        n_neg = (df["duration_sec"] < 0).sum()
        if n_neg > 0:
            log.warning("load_alerts: %d baris dengan duration_sec negatif.", n_neg)
    return df


# ══════════════════════════════════════════════════════════════════
# add_if_features  —  validasi 9 kolom fitur
# ══════════════════════════════════════════════════════════════════

FEATURE_COLS_V4 = [
    "alert_count",
    "max_severity",
    "duration_sec",
    "attacker_count",
    "rule_group_severity_enc",
    "agent_criticality",
    "hour_of_day",
    "unique_rules_triggered",   # f8
    "mitre_hit_count",          # f9
]


def add_if_features(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan / validasi 9 kolom fitur numerik untuk Isolation Forest.

    f1  alert_count
    f2  max_severity
    f3  duration_sec
    f4  attacker_count
    f5  rule_group_severity_enc
    f6  agent_criticality        (skala 1-4)
    f7  hour_of_day
    f8  unique_rules_triggered   [BARU]
    f9  mitre_hit_count          [BARU]
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

    # f8: unique_rules_triggered
    if "unique_rules_triggered" not in df.columns:
        log.warning(
            "Kolom unique_rules_triggered tidak ditemukan. "
            "Akan di-set 0. Pastikan run_rbta v4 yang menghasilkan CSV ini."
        )
        df["unique_rules_triggered"] = 0

    # f9: mitre_hit_count
    if "mitre_hit_count" not in df.columns:
        log.warning(
            "Kolom mitre_hit_count tidak ditemukan. "
            "Akan di-set 0. Pastikan has_mitre ada di Layer 1 (json_orches.py)."
        )
        df["mitre_hit_count"] = 0

    for col in FEATURE_COLS_V4:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["rule_groups"] = df["rule_groups"].astype(str).str.strip().str.lower()

    log.info("Feature vector 9-kolom siap. Statistik:")
    log.info("\n%s", df[FEATURE_COLS_V4].describe().round(2).to_string())

    return df


# ══════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing_01 import load_and_prepare

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/rbta_ready_all.csv"
    df_raw   = load_and_prepare(csv_path)

    df_meta, idx_map, elastic, wmark = run_rbta(
        df_raw,
        delta_t_minutes    = 15,
        max_window_minutes = 60,
        buffer_size        = 50,
        max_lateness_sec   = 10.0,
        enable_adaptive    = True,
    )

    df_meta = add_if_features(df_meta)

    n_neg = (df_meta["duration_sec"] < 0).sum()
    if n_neg > 0:
        log.error("FIX-1 GAGAL: masih ada %d duration_sec negatif.", n_neg)
        sys.exit(1)

    print(df_meta[FEATURE_COLS_V4].head(5).to_string())

    report = validate_mapping(df_raw, df_meta, idx_map)
    if not report["ok"]:
        log.error("Mapping tidak valid!")
        sys.exit(1)
    log.info("RBTA Enhanced v4 lulus validasi.")