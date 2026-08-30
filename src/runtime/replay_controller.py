from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from src.etl.wazuh_canonicalizer import canonicalize_wazuh_alert
from src.contracts.raw_alert import CanonicalRawAlert
from src.model.scoring_pipeline import ScoringPipeline
from src.runtime.durable_state import DurableStateManager
from src.runtime.escalation_sink import DeferredTelegramFileSink
from src.runtime.raw_evidence import RawAlertEvidenceStore
from src.runtime.service import LiveRBTAService

logger = logging.getLogger(__name__)

WITA_TIMEZONE = timezone(timedelta(hours=8))

SpeedFactor = Literal["1", "10", "100", "MAX"]
ReplayStatus = Literal["IDLE", "RUNNING", "PAUSED", "STOPPED", "COMPLETED", "ERROR"]

ALL_DATASETS_SENTINEL = "__ALL__"

class ReplayController:
    """Manages background replay runs deterministically with session isolation, strict canonicalization, and pacing."""

    def __init__(
        self,
        scoring_pipeline: ScoringPipeline,
        replay_data_dir: Optional[Union[str, Path]] = None,
        replay_runs_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.scoring_pipeline = scoring_pipeline

        if replay_data_dir is not None:
            self.data_dir = Path(replay_data_dir).resolve()
        else:
            self.data_dir = Path(os.environ.get("RBTA_REPLAY_DATA_DIR", "data/test_datasets")).resolve()

        if replay_runs_dir is not None:
            self.runs_dir = Path(replay_runs_dir).resolve()
        else:
            self.runs_dir = Path("data/runtime/replay-runs").resolve()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Current active run state
        self.run_id: Optional[str] = None
        self.status: ReplayStatus = "IDLE"
        self.dataset_name: Optional[str] = None
        self.dataset_mode: Literal["single", "all"] = "single"
        self.dataset_count: int = 1
        self.current_dataset: Optional[str] = None
        self.current_dataset_index: int = 0
        self.speed_factor: SpeedFactor = "MAX"
        self.processed_count: int = 0
        self.total_count: int = 0
        self.current_event_time: Optional[str] = None
        self.wall_clock_start: Optional[float] = None
        self.wall_clock_elapsed: float = 0.0
        self.last_error: Optional[Dict[str, Any]] = None

        self.current_service: Optional[LiveRBTAService] = None
        self.current_evidence_store: Optional[RawAlertEvidenceStore] = None
        self.escalation_sink: Optional[DeferredTelegramFileSink] = None

        # Telemetry, metrics, and trace ring buffer
        self._trace_buffer: deque = deque(maxlen=100)
        self.decision_counts: Dict[str, int] = {"ESCALATE": 0, "SUPPRESS": 0, "DAILY_DIGEST": 0}
        self._latest_scored_meta: Optional[Any] = None
        self._last_raw_alert_info: Optional[Dict[str, Any]] = None


    def list_datasets(self) -> List[Dict[str, Any]]:
        """List valid .jsonl replay datasets available in the data directory."""
        items: List[Dict[str, Any]] = []
        if not self.data_dir.exists():
            return items

        for p in sorted(self.data_dir.glob("*.jsonl")):
            if p.is_file():
                # Count lines efficiently
                line_count = 0
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                line_count += 1
                except Exception:
                    pass
                    
                items.append({
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "total_events": line_count,
                })
        return items

    def validate_dataset_path(self, dataset_name: str) -> Path:
        """Validate dataset name against directory traversal, absolute paths, and unsupported extensions."""
        if not dataset_name:
            raise ValueError("Dataset name cannot be empty")

        # Reject path separators and traversal
        if os.path.basename(dataset_name) != dataset_name or ".." in dataset_name or "/" in dataset_name or "\\" in dataset_name:
            raise ValueError(f"Path traversal detected in dataset_name: '{dataset_name}'")

        if not dataset_name.endswith(".jsonl"):
            raise ValueError(f"Replay datasets must be .jsonl files, got: '{dataset_name}'")

        target_path = (self.data_dir / dataset_name).resolve()
        # Verify it stays strictly within data_dir
        try:
            target_path.relative_to(self.data_dir)
        except ValueError:
            raise ValueError(f"Dataset path escapes data directory: '{dataset_name}'")

        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"Replay dataset not found: '{dataset_name}'")

        return target_path

    def _init_run_workspace(self, dataset_name: str, speed_factor: SpeedFactor) -> str:
        """Create a dedicated, isolated run directory with its own state and evidence databases."""
        new_run_id = str(uuid4())
        run_workspace = self.runs_dir / new_run_id
        run_workspace.mkdir(parents=True, exist_ok=True)

        state_mgr = DurableStateManager(run_workspace / "state.json")
        evidence_store = RawAlertEvidenceStore(run_workspace / "raw_alert_evidence.sqlite3")
        if "RBTA_TELEGRAM_PAYLOAD_PATH" in os.environ:
            telegram_sink_path = Path(os.environ["RBTA_TELEGRAM_PAYLOAD_PATH"]).resolve()
        else:
            telegram_sink_path = (self.runs_dir.parent / "telegram_escalate_payloads.txt").resolve()
        escalation_sink = DeferredTelegramFileSink(telegram_sink_path)

        service = LiveRBTAService(
            scoring_pipeline=self.scoring_pipeline,
            state_manager=state_mgr,
            raw_evidence_store=evidence_store,
            source_mode="REPLAY",
            escalation_sink=escalation_sink,
            run_id=new_run_id,
            auto_persist=False,
        )

        self.run_id = new_run_id
        self.dataset_name = dataset_name
        self.dataset_mode = "single"
        self.dataset_count = 1
        self.current_dataset = dataset_name
        self.current_dataset_index = 0
        self.speed_factor = speed_factor
        self.current_service = service
        self.current_evidence_store = evidence_store
        self.escalation_sink = escalation_sink
        self.processed_count = 0
        self.total_count = 0
        self.current_event_time = None
        self.wall_clock_start = None
        self.wall_clock_elapsed = 0.0
        self.last_error = None
        self.status = "IDLE"

        # Clear/initialize live telemetry
        self._trace_buffer.clear()
        self.decision_counts = {"ESCALATE": 0, "SUPPRESS": 0, "DAILY_DIGEST": 0}
        self._latest_scored_meta = None
        self._last_raw_alert_info = None

        self._persist_run_meta()
        return new_run_id

    def _model_version(self) -> str:
        """Extract exact model version from loaded scoring pipeline metadata fail-closed."""
        if self.scoring_pipeline is None or not self.scoring_pipeline.metadata:
            raise RuntimeError("Loaded scoring pipeline is missing mandatory model_version metadata")
        value = self.scoring_pipeline.metadata.get("model_version")
        if not value:
            raise RuntimeError("Loaded scoring pipeline is missing mandatory model_version metadata")
        return str(value)

    def _persist_run_meta(self) -> None:
        if not self.run_id:
            return
        meta_path = self.runs_dir / self.run_id / "run.json"
        data = {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "speed_factor": self.speed_factor,
            "status": self.status,
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "current_event_time": self.current_event_time,
            "wall_clock_elapsed_seconds": self.wall_clock_elapsed,
            "last_error": self.last_error,
            "model_version": self._model_version(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def start(self, dataset_name: str, speed_factor: SpeedFactor = "MAX") -> Dict[str, Any]:
        """Start a new deterministic replay stream on the specified dataset(s)."""
        with self._lock:
            if self.status in ("RUNNING", "PAUSED"):
                raise RuntimeError(f"Cannot start replay while status is '{self.status}'")

            if dataset_name in (ALL_DATASETS_SENTINEL, "ALL", "__ALL__"):
                datasets = [p for p in sorted(self.data_dir.glob("*.jsonl")) if p.is_file()]
                if not datasets:
                    raise ValueError("No replay datasets found in data directory")
                
                total = 0
                valid_datasets = []
                for p in datasets:
                    with open(p, "r", encoding="utf-8") as f:
                        file_has_lines = False
                        for line in f:
                            if line.strip():
                                total += 1
                                file_has_lines = True
                        if file_has_lines:
                            valid_datasets.append(p)
                
                if total == 0:
                    raise ValueError("No valid events found across any datasets")
                
                self._init_run_workspace(ALL_DATASETS_SENTINEL, speed_factor)
                self.dataset_mode = "all"
                self.dataset_count = len(valid_datasets)
                self.current_dataset = valid_datasets[0].name if valid_datasets else None
                self.current_dataset_index = 0
                dataset_paths = valid_datasets
            else:
                dataset_path = self.validate_dataset_path(dataset_name)

                # Count total non-empty lines in dataset
                total = 0
                with open(dataset_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            total += 1

                if total == 0:
                    raise ValueError(f"Replay dataset '{dataset_name}' contains no events")

                self._init_run_workspace(dataset_name, speed_factor)
                dataset_paths = [dataset_path]

            self.total_count = total
            self.status = "RUNNING"
            self.wall_clock_start = time.time()
            self._stop_event.clear()
            self._pause_event.set()

            self._thread = threading.Thread(
                target=self._run_loop,
                args=(dataset_paths,),
                daemon=True,
                name=f"replay-{self.run_id[:8]}",
            )
            self._thread.start()
            self._persist_run_meta()
            return self.get_status()

    def pause(self) -> Dict[str, Any]:
        """Pause wall-clock streaming progression without mutating event times or state."""
        with self._lock:
            if self.status != "RUNNING":
                return self.get_status()
            self._pause_event.clear()
            self.status = "PAUSED"
            if self.current_service:
                self.current_service.checkpoint()
            self._persist_run_meta()
            return self.get_status()

    def resume(self) -> Dict[str, Any]:
        """Resume streaming for the active paused run."""
        with self._lock:
            if self.status != "PAUSED":
                return self.get_status()
            self._pause_event.set()
            self.status = "RUNNING"
            self._persist_run_meta()
            return self.get_status()

    def stop(self) -> Dict[str, Any]:
        """Stop replay immediately. Does NOT force-drain active buckets, preserving partial state."""
        with self._lock:
            if self.status not in ("RUNNING", "PAUSED"):
                return self.get_status()

            self._stop_event.set()
            self._pause_event.set()  # Unblock if paused
            self.status = "STOPPED"
            if self.current_service:
                self.current_service.checkpoint()
            self._persist_run_meta()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        return self.get_status()

    def reset(self) -> Dict[str, Any]:
        """Reset replay controller and prepare a clean new run workspace."""
        with self._lock:
            if self.status in ("RUNNING", "PAUSED"):
                self.stop()

            self.run_id = None
            self.status = "IDLE"
            self.dataset_name = None
            self.dataset_mode = "single"
            self.dataset_count = 1
            self.current_dataset = None
            self.current_dataset_index = 0
            self.processed_count = 0
            self.total_count = 0
            self.current_event_time = None
            self.wall_clock_start = None
            self.wall_clock_elapsed = 0.0
            self.last_error = None
            self.current_service = None
            self.current_evidence_store = None
            return self.get_status()

    def wait_until_complete(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Wait until active replay run reaches a terminal state (COMPLETED, STOPPED, ERROR)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self.status in ("COMPLETED", "STOPPED", "ERROR"):
                    return self.get_status()
            time.sleep(0.05)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self.get_status()

    def _run_loop(self, dataset_paths: List[Path]) -> None:
        """Internal replay worker thread processing canonical alerts strictly."""
        last_event_ts: Optional[float] = None
        speed_multiplier = float(self.speed_factor) if self.speed_factor != "MAX" else None
        last_checkpoint_count = 0
        last_checkpoint_time = time.time()

        try:
            for idx, dataset_path in enumerate(dataset_paths):
                with self._lock:
                    self.current_dataset = dataset_path.name
                    self.current_dataset_index = idx

                line_no = 0
                with open(dataset_path, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        line_no += 1
                        line_str = raw_line.strip()
                        if not line_str:
                            continue

                        # Check stop signal
                        if self._stop_event.is_set():
                            break

                        # Check pause signal (blocks until resumed or stopped)
                        while not self._pause_event.is_set():
                            if self._stop_event.is_set():
                                break
                            time.sleep(0.02)

                        if self._stop_event.is_set():
                            break

                        # Parse JSON
                        try:
                            raw_data = json.loads(line_str)
                        except Exception as e:
                            raise ValueError(f"Malformed JSON on line {line_no}: {e}") from e

                        # Canonicalize strictly through single canonical path
                        canonical_alert = canonicalize_wazuh_alert(raw_data)

                        # Pacing if not MAX speed
                        event_ts_sec = canonical_alert.timestamp.timestamp()
                        if speed_multiplier is not None and last_event_ts is not None:
                            dt = (event_ts_sec - last_event_ts) / speed_multiplier
                            if dt > 0:
                                sleep_end = time.monotonic() + min(dt, 5.0)
                                while time.monotonic() < sleep_end:
                                    if self._stop_event.is_set():
                                        break
                                    if not self._pause_event.is_set():
                                        break
                                    time.sleep(0.01)

                        if self._stop_event.is_set():
                            break
                        while not self._pause_event.is_set():
                            if self._stop_event.is_set():
                                break
                            time.sleep(0.02)
                        if self._stop_event.is_set():
                            break

                        last_event_ts = event_ts_sec

                        # Ingest alert into isolated run service
                        assert self.current_service is not None
                        scored_metas = self.current_service.ingest_alert(canonical_alert)
                        now_ts = datetime.now(WITA_TIMEZONE).strftime("%H:%M:%S.%f")[:-3]

                        with self._lock:
                            self.processed_count += 1
                            self.current_event_time = canonical_alert.timestamp.isoformat()
                            self._last_raw_alert_info = {
                                "alert_id": canonical_alert.wazuh_alert_id,
                                "agent_id": canonical_alert.agent_id,
                                "agent_name": canonical_alert.agent_name,
                                "rule_group": canonical_alert.rule_group_primary,
                                "rule_id": canonical_alert.rule_id,
                                "level": canonical_alert.rule_level,
                                "timestamp": canonical_alert.timestamp.isoformat(),
                            }

                            if self.processed_count % 30 == 1:
                                self._trace_buffer.append({
                                    "timestamp": now_ts,
                                    "stage": "RAW",
                                    "message": f"Alert {canonical_alert.wazuh_alert_id} received",
                                    "detail": f"Agent {canonical_alert.agent_id} ({canonical_alert.agent_name}) | {canonical_alert.rule_group_primary}",
                                })
                                self._trace_buffer.append({
                                    "timestamp": now_ts,
                                    "stage": "CANONICAL",
                                    "message": "CanonicalRawAlert generated",
                                    "detail": f"Rule {canonical_alert.rule_id} (Level {canonical_alert.rule_level})",
                                })

                            if scored_metas:
                                for sm in scored_metas:
                                    self._latest_scored_meta = sm
                                    action = sm.action if sm.action in self.decision_counts else "SUPPRESS"
                                    self.decision_counts[action] = self.decision_counts.get(action, 0) + 1

                                    self._trace_buffer.append({
                                        "timestamp": now_ts,
                                        "stage": "FINALIZE",
                                        "message": f"MetaAlert #{sm.meta_id} finalized",
                                        "detail": f"{sm.alert_count} raw alerts -> MetaAlert ({sm.rule_group_primary})",
                                    })
                                    self._trace_buffer.append({
                                        "timestamp": now_ts,
                                        "stage": "FEATURES",
                                        "message": "Seven-feature vector extracted",
                                        "detail": f"Max sev {sm.max_severity}, tactics {len(sm.mitre_tactics)}",
                                    })
                                    self._trace_buffer.append({
                                        "timestamp": now_ts,
                                        "stage": "SCORE",
                                        "message": "Isolation Forest scored",
                                        "detail": f"Anomaly score {sm.anomaly_score:.6f} (raw {sm.raw_model_score:.6f})",
                                    })
                                    self._trace_buffer.append({
                                        "timestamp": now_ts,
                                        "stage": "DECISION",
                                        "message": f"{sm.decision} -> {sm.action}",
                                        "detail": f"Threshold {sm.threshold_used:.6f} (margin {sm.anomaly_score - sm.threshold_used:+.6f})",
                                    })
                                    if sm.action == "ESCALATE":
                                        self._trace_buffer.append({
                                            "timestamp": now_ts,
                                            "stage": "OUTPUT",
                                            "message": "Deferred Telegram payload written",
                                            "detail": f"MetaAlert #{sm.meta_id} ({sm.decision})",
                                        })

                            if self.wall_clock_start is not None:
                                self.wall_clock_elapsed = max(0.0, time.time() - self.wall_clock_start)

                        # Periodic durable checkpoint (every 500 events or every 1.0s)
                        now_wall = time.time()
                        if (self.processed_count - last_checkpoint_count >= 500) or (now_wall - last_checkpoint_time >= 1.0):
                            if self.current_service:
                                self.current_service.checkpoint()
                            with self._lock:
                                self._persist_run_meta()
                            last_checkpoint_count = self.processed_count
                            last_checkpoint_time = now_wall
                
                if self._stop_event.is_set():
                    break

            # Check if finished naturally (EOF reached without manual stop)
            if not self._stop_event.is_set():
                # Drain remaining active buckets at natural EOF
                if self.current_service:
                    drained_metas = self.current_service.shutdown(drain=True)
                    if drained_metas:
                        with self._lock:
                            now_ts = datetime.now(WITA_TIMEZONE).strftime("%H:%M:%S.%f")[:-3]
                            for sm in drained_metas:
                                self._latest_scored_meta = sm
                                action = sm.action if sm.action in self.decision_counts else "SUPPRESS"
                                self.decision_counts[action] = self.decision_counts.get(action, 0) + 1
                                self._trace_buffer.append({
                                    "timestamp": now_ts,
                                    "stage": "FINALIZE",
                                    "message": f"MetaAlert #{sm.meta_id} finalized (drain)",
                                    "detail": f"{sm.alert_count} raw alerts -> MetaAlert ({sm.rule_group_primary})",
                                })
                                self._trace_buffer.append({
                                    "timestamp": now_ts,
                                    "stage": "DECISION",
                                    "message": f"{sm.decision} -> {sm.action}",
                                    "detail": f"Anomaly {sm.anomaly_score:.6f} vs threshold {sm.threshold_used:.6f}",
                                })
                                if sm.action == "ESCALATE":
                                    self._trace_buffer.append({
                                        "timestamp": now_ts,
                                        "stage": "OUTPUT",
                                        "message": "Deferred Telegram payload written",
                                        "detail": f"MetaAlert #{sm.meta_id} ({sm.decision})",
                                    })

                if self.current_evidence_store:
                    self.current_evidence_store.flush()

                with self._lock:
                    self.status = "COMPLETED"
                    if self.wall_clock_start is not None:
                        self.wall_clock_elapsed = max(0.0, time.time() - self.wall_clock_start)
                    self._persist_run_meta()

        except Exception as e:
            logger.exception("Replay failed on dataset '%s', line %d", self.current_dataset, line_no)
            if self.current_service:
                try:
                    self.current_service.checkpoint()
                except Exception:
                    pass
            with self._lock:
                self.status = "ERROR"
                self.last_error = {
                    "dataset": self.current_dataset,
                    "line_number": line_no,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                if self.wall_clock_start is not None:
                    self.wall_clock_elapsed = max(0.0, time.time() - self.wall_clock_start)
                self._persist_run_meta()

    def get_status(self) -> Dict[str, Any]:
        """Return standardized status DTO with comprehensive pipeline telemetry for the replay controller."""
        with self._lock:
            elapsed = self.wall_clock_elapsed
            if self.status == "RUNNING" and self.wall_clock_start is not None:
                elapsed = max(0.0, time.time() - self.wall_clock_start)

            eps = (self.processed_count / elapsed) if elapsed > 0 else 0.0
            progress = (self.processed_count / self.total_count) if self.total_count > 0 else 0.0
            model_ver = getattr(self.scoring_pipeline.bundle, "model_version", "v1") if self.scoring_pipeline and hasattr(self.scoring_pipeline, "bundle") else "v1"

            latest_meta_dict = None
            if self._latest_scored_meta:
                sm = self._latest_scored_meta
                latest_meta_dict = {
                    "meta_id": sm.meta_id,
                    "agent_id": sm.agent_id,
                    "agent_name": sm.agent_name,
                    "rule_group_primary": sm.rule_group_primary,
                    "start_time": sm.start_time.isoformat() if sm.start_time else None,
                    "end_time": sm.end_time.isoformat() if sm.end_time else None,
                    "alert_count": sm.alert_count,
                    "max_severity": sm.max_severity,
                    "mitre_tactics": list(sm.mitre_tactics),
                    "seven_features": dict(sm.seven_features) if sm.seven_features else {},
                    "raw_model_score": round(float(sm.raw_model_score), 6),
                    "anomaly_score": round(float(sm.anomaly_score), 6),
                    "threshold_used": round(float(sm.threshold_used), 6),
                    "margin": round(float(sm.anomaly_score - sm.threshold_used), 6),
                    "decision": sm.decision,
                    "action": sm.action,
                    "escalate": sm.escalate,
                    "model_version": sm.model_version,
                }

            active_buckets = 0
            active_agents = 0
            finalized_count = 0
            if self.current_service:
                active_buckets = len(self.current_service.engine._active_buckets) if hasattr(self.current_service.engine, "_active_buckets") else 0
                active_agents = len(self.current_service.engine._temporal_states) if hasattr(self.current_service.engine, "_temporal_states") else 0
                finalized_count = len(self.current_service.finalized_history)

            evidence_count = self.current_evidence_store.count() if self.current_evidence_store else self.processed_count
            telegram_count = self.escalation_sink.get_total_count() if self.escalation_sink else 0
            latest_payloads = self.escalation_sink.get_latest_payloads(1) if self.escalation_sink else []
            latest_payload = latest_payloads[0] if latest_payloads else None

            telemetry = {
                "raw": {
                    "processed": self.processed_count,
                    "evidence_count": evidence_count,
                    "last_alert": self._last_raw_alert_info,
                },
                "rbta": {
                    "active_buckets": active_buckets,
                    "finalized_meta_alerts": finalized_count,
                    "active_agents": active_agents,
                },
                "latest_meta_alert": latest_meta_dict,
                "decision_counts": dict(self.decision_counts),
                "output": {
                    "telegram_deferred_count": telegram_count,
                    "latest_payload": latest_payload,
                },
                "trace": list(self._trace_buffer)[-50:],
            }

            return {
                "run_id": self.run_id,
                "status": self.status,
                "dataset": self.dataset_name,
                "dataset_mode": self.dataset_mode,
                "dataset_count": self.dataset_count,
                "current_dataset": self.current_dataset,
                "current_dataset_index": self.current_dataset_index,
                "speed": self.speed_factor,
                "processed_count": self.processed_count,
                "total_count": self.total_count,
                "progress": round(progress, 4),
                "current_event_time": self.current_event_time,
                "wall_clock_elapsed_seconds": round(elapsed, 2),
                "events_per_second": round(eps, 2),
                "model_version": model_ver,
                "error": self.last_error.get("error_message") if self.last_error else None,
                "last_error": self.last_error,
                "telemetry": telemetry,
            }

    def get_telegram_payloads(self, limit: int = 50) -> Dict[str, Any]:
        """Return latest deferred Telegram payloads and total count."""
        with self._lock:
            if not self.escalation_sink:
                if "RBTA_TELEGRAM_PAYLOAD_PATH" in os.environ:
                    telegram_sink_path = Path(os.environ["RBTA_TELEGRAM_PAYLOAD_PATH"]).resolve()
                else:
                    telegram_sink_path = (self.runs_dir.parent / "telegram_escalate_payloads.txt").resolve()
                sink = DeferredTelegramFileSink(telegram_sink_path)
                return {
                    "items": sink.get_latest_payloads(limit=limit),
                    "total_count": sink.get_total_count(),
                }
            return {
                "items": self.escalation_sink.get_latest_payloads(limit=limit),
                "total_count": self.escalation_sink.get_total_count(),
            }
