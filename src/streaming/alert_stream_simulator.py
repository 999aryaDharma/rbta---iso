"""
alert_stream_simulator.py  —  Real-time Alert Stream Emulator
==============================================================
Simulate online alert streaming dari static CSV dataset dengan:
  - Time-accurate replay dengan acceleration control
  - Informatif logging untuk monitoring simulasi
  - Callback interface untuk real-time processing
  - Statistik lengkap tentang stream content

Usage:
  sim = AlertStreamSimulator(
      csv_path="data/raw/rbta_ready_all.csv",
      speed_factor=100
  )
  sim.on_alert(lambda alert: process(alert))
  sim.replay()
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StreamStats:
    """Statistics collector untuk streaming simulation."""
    total_alerts: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    agent_counts: dict = field(default_factory=dict)
    rule_group_counts: dict = field(default_factory=dict)
    severity_counts: dict = field(default_factory=dict)
    mitre_coverage: int = 0
    
    def add_alert(self, alert: pd.Series) -> None:
        """Update stats dengan alert baru."""
        self.total_alerts += 1
        
        # Agent distribution
        agent = str(alert.get("agent_name", "unknown")).strip()
        self.agent_counts[agent] = self.agent_counts.get(agent, 0) + 1
        
        # Rule group distribution
        rg = str(alert.get("rule_group_primary", "unknown")).strip()
        self.rule_group_counts[rg] = self.rule_group_counts.get(rg, 0) + 1
        
        # Severity distribution
        sev = int(alert.get("rule_level", 1))
        sev_bucket = self._severity_bucket(sev)
        self.severity_counts[sev_bucket] = self.severity_counts.get(sev_bucket, 0) + 1
        
        # MITRE coverage
        if pd.notna(alert.get("mitre_tactic")) and str(alert.get("mitre_tactic")).strip():
            self.mitre_coverage += 1
    
    @staticmethod
    def _severity_bucket(level: int) -> str:
        """Bucket severity ke kategori."""
        if level <= 2:
            return "1-2_Info"
        elif level <= 4:
            return "3-4_Medium"
        elif level <= 6:
            return "5-6_High"
        else:
            return "7+_Critical"
    
    @property
    def duration_sec(self) -> float:
        """Duration simulasi dalam detik."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def alerts_per_sec(self) -> float:
        """Throughput (alerts/sec)."""
        dur = self.duration_sec
        if dur > 0:
            return self.total_alerts / dur
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Logger Setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_stream_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """Setup logger untuk streaming simulation."""
    logger = logging.getLogger("AlertStreamSimulator")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (jika specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, log_level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Logging to file: %s", log_file)
    
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Main Simulator
# ══════════════════════════════════════════════════════════════════════════════

class AlertStreamSimulator:
    """
    Real-time alert streaming simulator dari static CSV.
    
    Features:
      - Load CSV dan sort by timestamp
      - Replay dengan time-accurate delays (accelerated)
      - Callback interface untuk setiap alert
      - Informatif logging dengan progress tracking
      - Statistical summary di akhir
    """
    
    def __init__(
        self,
        csv_path: str = "data/raw/rbta_ready_all.csv",
        speed_factor: float = 100.0,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        log_frequency: int = 1000,
        filter_agents: Optional[list[str]] = None,
        filter_severity_min: Optional[int] = None,
        export_jsonl: bool = False,
        export_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path ke raw CSV alert data
        speed_factor : float
            Acceleration multiplier (100 = 100x kecepatan)
        log_file : str, optional
            Path untuk output log file
        log_level : str
            Log level (DEBUG, INFO, WARNING, ERROR)
        log_frequency : int
            Log progress setiap N alerts
        filter_agents : list, optional
            Jika set, hanya replay alerts dari agent ini
        filter_severity_min : int, optional
            Jika set, hanya replay alerts dengan severity >= ini
        export_jsonl : bool
            Export alerts ke JSONL format alongside
        export_path : str, optional
            Path untuk exported JSONL (jika export_jsonl=True)
        """
        self.csv_path = csv_path
        self.speed_factor = speed_factor
        self.log_frequency = log_frequency
        self.filter_agents = filter_agents
        self.filter_severity_min = filter_severity_min
        self.export_jsonl = export_jsonl
        self.export_path = export_path or "output/alerts_stream.jsonl"
        
        self.logger = setup_stream_logger(log_file, log_level)
        self.df = None
        self.callback = None
        self.stats = StreamStats()
        self._export_file = None
    
    def load_data(self) -> int:
        """Load dan preprocess CSV data."""
        self.logger.info("Loading CSV data: %s", self.csv_path)
        
        if not os.path.exists(self.csv_path):
            self.logger.error("CSV file not found: %s", self.csv_path)
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        file_size_mb = os.path.getsize(self.csv_path) / (1024 ** 2)
        self.logger.info("File size: %.2f MB", file_size_mb)
        
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            self.logger.error("Failed to load CSV: %s", e)
            raise
        
        # Parse timestamps
        try:
            self.df["timestamp_utc"] = pd.to_datetime(
                self.df["timestamp_utc"], errors="coerce"
            )
        except Exception as e:
            self.logger.warning("Failed to parse timestamps: %s", e)
            self.df["timestamp_utc"] = pd.NaT
        
        # Sort by timestamp
        self.df = self.df.sort_values("timestamp_utc").reset_index(drop=True)
        
        n_raw = len(self.df)
        self.logger.info("Loaded %d raw alerts", n_raw)
        
        # Apply filters
        if self.filter_agents:
            mask_agent = self.df["agent_name"].isin(self.filter_agents)
            self.df = self.df[mask_agent]
            self.logger.info("  Filter agents=%s: %d alerts remain", 
                           self.filter_agents, len(self.df))
        
        if self.filter_severity_min is not None:
            mask_sev = self.df["rule_level"] >= self.filter_severity_min
            self.df = self.df[mask_sev]
            self.logger.info("  Filter severity >= %d: %d alerts remain",
                           self.filter_severity_min, len(self.df))
        
        n_filtered = len(self.df)
        if n_filtered < n_raw:
            self.logger.info("After filtering: %d alerts (%.1f%% of original)",
                           n_filtered, n_filtered / n_raw * 100)
        
        # Date range info
        t_min = self.df["timestamp_utc"].min()
        t_max = self.df["timestamp_utc"].max()
        if pd.notna(t_min) and pd.notna(t_max):
            duration_min = (t_max - t_min).total_seconds() / 60
            self.logger.info("Date range: %s to %s (%.1f minutes)", 
                           t_min, t_max, duration_min)
            sim_duration_sec = duration_min * 60 / self.speed_factor
            self.logger.info("Simulation will take ~%.1f seconds at %.0fx speed",
                           sim_duration_sec, self.speed_factor)
        
        # Agent & rule_group diversity
        n_agents = self.df["agent_name"].nunique()
        n_rulegroups = self.df["rule_group_primary"].nunique()
        self.logger.info("Alert diversity: %d unique agents, %d unique rule groups",
                       n_agents, n_rulegroups)
        
        return n_filtered
    
    def on_alert(self, callback: Callable[[pd.Series], None]) -> None:
        """Register callback untuk setiap alert yang di-replay."""
        self.callback = callback
        self.logger.debug("Callback registered: %s", callback.__name__)
    
    def replay(self) -> StreamStats:
        """Main simulation loop — replay semua alerts dengan time-accurate delays."""
        if self.df is None:
            n = self.load_data()
        else:
            n = len(self.df)
        
        if n == 0:
            self.logger.warning("No alerts to replay after filtering")
            return self.stats
        
        # Setup export
        if self.export_jsonl:
            os.makedirs(os.path.dirname(self.export_path) or ".", exist_ok=True)
            self._export_file = open(self.export_path, "w", encoding="utf-8")
            self.logger.info("Exporting to JSONL: %s", self.export_path)
        
        # Print header
        self.logger.info("=" * 80)
        self.logger.info("  ALERT STREAM SIMULATION — STARTING")
        self.logger.info("=" * 80)
        self.logger.info("Total alerts: %d", n)
        self.logger.info("Speed factor: %.0fx", self.speed_factor)
        self.logger.info("Callbacks: %s", "enabled" if self.callback else "disabled")
        self.logger.info("-" * 80)
        
        self.stats.start_time = datetime.now()
        t_base = self.df.iloc[0]["timestamp_utc"]
        t_start_wall = time.perf_counter()
        last_log_idx = 0
        last_log_wall = t_start_wall
        
        try:
            for idx, (_, alert) in enumerate(self.df.iterrows(), 1):
                # Calculate delay
                t_alert = alert.get("timestamp_utc")
                if pd.notna(t_alert) and pd.notna(t_base):
                    delay_sec = (t_alert - t_base).total_seconds() / self.speed_factor
                    elapsed_wall = time.perf_counter() - t_start_wall
                    sleep_time = delay_sec - elapsed_wall
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Callback
                if self.callback:
                    try:
                        self.callback(alert)
                    except Exception as e:
                        self.logger.error("Callback failed for alert %d: %s", idx, e)
                
                # Export
                if self._export_file:
                    try:
                        alert_dict = alert.to_dict()
                        # Convert non-serializable types
                        for k, v in alert_dict.items():
                            if pd.isna(v):
                                alert_dict[k] = None
                            elif isinstance(v, (pd.Timestamp, datetime)):
                                alert_dict[k] = str(v)
                        
                        import json
                        self._export_file.write(json.dumps(alert_dict) + "\n")
                    except Exception as e:
                        self.logger.warning("Failed to export alert %d: %s", idx, e)
                
                # Update stats
                self.stats.add_alert(alert)
                
                # Logging & progress
                if idx % self.log_frequency == 0 or idx == n:
                    t_now_wall = time.perf_counter()
                    elapsed_wall = t_now_wall - t_start_wall
                    rate = (idx - last_log_idx) / (t_now_wall - last_log_wall)
                    
                    pct = idx / n * 100
                    eta_sec = (n - idx) / rate if rate > 0 else 0
                    
                    self.logger.info(
                        "[PROGRESS] %d / %d alerts (%.1f%%) | %.0f alerts/sec | ETA: %.1fs",
                        idx, n, pct, rate, eta_sec
                    )
                    
                    # Agent distribution snapshot
                    top_agents = sorted(
                        self.stats.agent_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    agent_str = " | ".join([f"{a}:{c}" for a, c in top_agents])
                    self.logger.debug("  Top agents: %s", agent_str)
                    
                    # Rule group snapshot
                    top_rulegroups = sorted(
                        self.stats.rule_group_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    rg_str = " | ".join([f"{rg}:{c}" for rg, c in top_rulegroups])
                    self.logger.debug("  Top rule groups: %s", rg_str)
                    
                    last_log_idx = idx
                    last_log_wall = t_now_wall
        
        finally:
            if self._export_file:
                self._export_file.close()
                self.logger.info("Export file closed: %s", self.export_path)
        
        self.stats.end_time = datetime.now()
        
        # Print summary
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self) -> None:
        """Print detailed summary di akhir replay."""
        self.logger.info("=" * 80)
        self.logger.info("  ALERT STREAM SIMULATION — COMPLETED")
        self.logger.info("=" * 80)
        
        self.logger.info("Total alerts processed: %d", self.stats.total_alerts)
        self.logger.info("Wall-clock duration: %.1f seconds", self.stats.duration_sec)
        self.logger.info("Processing rate: %.0f alerts/sec", self.stats.alerts_per_sec)
        self.logger.info("")
        
        # Agent distribution
        self.logger.info("Agent distribution:")
        for agent, count in sorted(
            self.stats.agent_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = count / self.stats.total_alerts * 100
            self.logger.info("  %-20s: %6d (%.1f%%)", agent, count, pct)
        
        self.logger.info("")
        
        # Rule group distribution
        self.logger.info("Alert type distribution (rule_group):")
        for rg, count in sorted(
            self.stats.rule_group_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = count / self.stats.total_alerts * 100
            self.logger.info("  %-20s: %6d (%.1f%%)", rg, count, pct)
        
        self.logger.info("")
        
        # Severity distribution
        self.logger.info("Severity distribution:")
        for sev_bucket, count in sorted(self.stats.severity_counts.items()):
            pct = count / self.stats.total_alerts * 100
            self.logger.info("  %-15s: %6d (%.1f%%)", sev_bucket, count, pct)
        
        self.logger.info("")
        
        # MITRE coverage
        mitre_pct = self.stats.mitre_coverage / self.stats.total_alerts * 100
        self.logger.info("MITRE ATT&CK coverage: %d alerts (%.1f%%)",
                       self.stats.mitre_coverage, mitre_pct)
        
        self.logger.info("=" * 80)
    
    def get_statistics(self) -> StreamStats:
        """Return collected statistics."""
        return self.stats


# ══════════════════════════════════════════════════════════════════════════════
# Example / Testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example 1: Basic replay dengan logging
    sim = AlertStreamSimulator(
        csv_path="data/raw/rbta_ready_all.csv",
        speed_factor=100,
        log_file="output/stream_simulation.log",
        log_level="INFO",
        log_frequency=5000,
    )
    
    # Simple callback untuk count
    alert_count = [0]
    def simple_callback(alert):
        alert_count[0] += 1
    
    sim.on_alert(simple_callback)
    stats = sim.replay()
    
    print("\n" + "=" * 80)
    print(f"Simulation complete: {stats.total_alerts} alerts in {stats.duration_sec:.1f}s")
    print("=" * 80)
