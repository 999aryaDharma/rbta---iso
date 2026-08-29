"""Deterministic replay controller managing background replay runs with speed control and session isolation."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from src.contracts.raw_alert import CanonicalRawAlert
from src.runners.clock import ReplayClock
from src.runners.replay_runner import ReplayStreamRunner
from src.model.scoring_pipeline import ScoringPipeline


class ReplayController:
    """Manages background replay runs deterministically."""

    def __init__(
        self,
        scoring_pipeline: ScoringPipeline,
        raw_evidence_store: Any = None,
        replay_data_dir: Optional[Path] = None,
        replay_runs_dir: Optional[Path] = None,
    ) -> None:
        self.scoring_pipeline = scoring_pipeline
        self.raw_evidence_store = raw_evidence_store
        self.run_id: Optional[str] = None
        self.status: str = "IDLE"
        self.processed_count: int = 0
        self.total_count: int = 0
        self.current_event_time: Optional[str] = None
        self.wall_clock_start: Optional[float] = None
        self.speed: float = 1.0

        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Mount directory from env or argument
        if replay_data_dir is not None:
            self.data_dir = Path(replay_data_dir).resolve()
        else:
            self.data_dir = Path(os.environ.get("RBTA_REPLAY_DATA_DIR", "data/test_datasets")).resolve()

        if replay_runs_dir is not None:
            self.runs_dir = Path(replay_runs_dir).resolve()
        else:
            self.runs_dir = Path("data/runtime/replay-runs").resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _read_dataset(self) -> List[CanonicalRawAlert]:
        """Read alerts from the dataset directory (JSONL files)."""
        alerts: List[CanonicalRawAlert] = []
        if not self.data_dir.exists():
            return alerts

        for filepath in sorted(self.data_dir.glob("*.jsonl")):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "timestamp" in data:
                            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                            if data["timestamp"].tzinfo is None:
                                data["timestamp"] = data["timestamp"].replace(tzinfo=timezone.utc)
                        alert = CanonicalRawAlert(
                            wazuh_alert_id=str(data.get("wazuh_alert_id", "")),
                            timestamp=data["timestamp"],
                            agent_id=str(data.get("agent_id", "001")),
                            agent_name=str(data.get("agent_name", "unknown")),
                            rule_group_primary=str(data.get("rule_group_primary", "unknown")),
                            rule_level=int(data.get("rule_level", 3)),
                            rule_id=str(data.get("rule_id", "1000")),
                            mitre_tactics=tuple(data.get("mitre_tactics", ())),
                            srcip=data.get("srcip"),
                            agent_criticality=int(data.get("agent_criticality", 1)),
                            metadata=data.get("metadata", {}),
                        )
                        alerts.append(alert)
                    except Exception:
                        pass
        return alerts

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = 0.0
            if self.wall_clock_start is not None:
                elapsed = max(0.0, time.time() - self.wall_clock_start)
            events_per_sec = (self.processed_count / elapsed) if elapsed > 0 else 0.0
            return {
                "run_id": self.run_id,
                "status": self.status,
                "dataset": self.data_dir.name if self.data_dir.exists() else None,
                "processed_count": self.processed_count,
                "total_count": self.total_count,
                "current_event_time": self.current_event_time,
                "wall_clock_elapsed_seconds": elapsed,
                "wall_clock_start": self.wall_clock_start,
                "speed": self.speed,
                "events_per_second": events_per_sec,
            }

    def start(self, speed: float = 1.0, dataset: Optional[str] = None) -> None:
        with self._lock:
            if self.status in ("RUNNING", "PAUSED"):
                return

            self.run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.status = "RUNNING"
            self.processed_count = 0
            self.total_count = 0
            self.speed = speed
            self.wall_clock_start = time.time()

            self._stop_event.clear()
            self._pause_event.set()

            self._thread = threading.Thread(target=self._run_worker, daemon=True)
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self.status == "RUNNING":
                self.status = "PAUSED"
                self._pause_event.clear()

    def resume(self) -> None:
        with self._lock:
            if self.status == "PAUSED":
                self.status = "RUNNING"
                self._pause_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            self.status = "COMPLETED"

    def reset(self) -> None:
        self.stop()
        with self._lock:
            self.run_id = None
            self.status = "IDLE"
            self.processed_count = 0
            self.total_count = 0
            self.current_event_time = None
            self.wall_clock_start = None

    def _run_worker(self) -> None:
        try:
            alerts = self._read_dataset()
            with self._lock:
                self.total_count = len(alerts)

            run_out_dir = self.runs_dir / (self.run_id or "unknown")
            run_out_dir.mkdir(parents=True, exist_ok=True)
            out_file_path = run_out_dir / "meta_alerts.jsonl"

            clock_speed = "MAX" if (self.speed <= 0 or self.speed == float("inf")) else self.speed
            clock = ReplayClock(speed_factor=clock_speed)
            runner = ReplayStreamRunner(
                scoring_pipeline=self.scoring_pipeline,
                clock=clock,
                adaptive=True,
            )

            def alert_generator():
                for alert in alerts:
                    if self._stop_event.is_set():
                        break

                    while not self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.01)

                    if self._stop_event.is_set():
                        break

                    with self._lock:
                        self.processed_count += 1
                        self.current_event_time = alert.timestamp.isoformat()

                    if self.raw_evidence_store:
                        self.raw_evidence_store.store(alert, source_mode="REPLAY")

                    yield alert

            with open(out_file_path, "w", encoding="utf-8") as out_file:
                for meta in runner.run(alert_generator()):
                    out_file.write(
                        json.dumps({
                            "meta_id": meta.meta_id,
                            "anomaly_score": meta.anomaly_score,
                            "decision": meta.decision,
                            "action": meta.action,
                        })
                        + "\n"
                    )
                    out_file.flush()

            with self._lock:
                if not self._stop_event.is_set():
                    self.status = "COMPLETED"

        except Exception as exc:
            with self._lock:
                self.status = "ERROR"
