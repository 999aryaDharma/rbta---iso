"""Live worker thread running ingestion cycles inside the FastAPI process (L2).

Design decisions (locked 2026-09-29):
- The worker is a single daemon thread owned by the backend process lifespan.
  No separate process, no IPC; the service RLock already serializes mutations.
- Constraint: exactly one backend process. Multi-worker uvicorn would start
  one thread per process and is NOT supported (single-writer interprocess
  lock is an explicit non-goal).
- Disabled by default (RBTA_LIVE_WORKER_ENABLED=true to opt in) so normal
  replay/demo operation is unaffected.
"""

from datetime import datetime, timezone
import logging
import os
import random
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from src.runtime.durable_state import compute_derivation_hash
from src.runtime.service import LiveRBTAService

logger = logging.getLogger(__name__)

_JOIN_TIMEOUT_SEC = 30.0

# Upper bound for exponential backoff between failed ingestion cycles.
_BACKOFF_MAX_SEC = 300.0

# Process-lifetime handles for OS-level state locks (held until the process
# exits to guarantee a single live writer; released early only in tests via
# release_state_lock).
_STATE_LOCK_HANDLES: list = []

# Lock-file path (str) -> open handle, backing release_state_lock.
_STATE_LOCK_PATHS: Dict[str, Any] = {}


def worker_enabled_from_env(env_map: Mapping[str, str]) -> bool:
    """Whether the live worker thread is opted in via environment."""
    return str(env_map.get("RBTA_LIVE_WORKER_ENABLED", "false")).strip().lower() == "true"


def drain_on_stop_from_env(env_map: Mapping[str, str]) -> bool:
    """Whether worker shutdown drains open buckets (default false).

    Drain is an explicit decommission operation, not the normal shutdown
    path: on plain restart open buckets stay preserved in durable state.
    """
    return str(env_map.get("RBTA_LIVE_WORKER_DRAIN_ON_STOP", "false")).strip().lower() == "true"


def acquire_state_lock(state_path: Any):
    """Hold an OS-level exclusive lock for ``<state>.lock`` (fail fast).

    The lock is held for the lifetime of the process (handles are retained
    in :data:`_STATE_LOCK_HANDLES` and never released, except in tests via
    :func:`release_state_lock`) so that at most one backend process ever
    writes the live runtime state.

    Raises
    ------
    RuntimeError
        If another live writer already holds the lock.
    """
    lock_path = Path(str(state_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        handle = open(lock_path, "a+b")
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\x00")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"Live state '{state_path}' is locked: another live writer holds "
                    f"'{lock_path}'. Refusing to start a second live writer."
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"Live state '{state_path}' is locked: another live writer holds "
                    f"'{lock_path}'. Refusing to start a second live writer."
                ) from exc
    except BaseException:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        raise
    _STATE_LOCK_HANDLES.append(handle)
    _STATE_LOCK_PATHS[str(lock_path)] = handle
    return handle


def release_state_lock(state_path: Any) -> bool:
    """Best-effort release of a lock held via :func:`acquire_state_lock`.

    Unlocks, closes, and forgets the registry handle for ``<state>.lock``.
    Production never calls this (the lock is process-lifetime); it exists
    so tests can release and re-acquire. Never raises: returns True when a
    held lock was released, False when nothing was held.
    """
    lock_path = str(Path(str(state_path) + ".lock"))
    handle = _STATE_LOCK_PATHS.pop(lock_path, None)
    if handle is None:
        return False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                handle.close()
            except OSError:
                pass
    finally:
        try:
            _STATE_LOCK_HANDLES.remove(handle)
        except ValueError:
            pass
    return True


def poll_interval_from_env(env_map: Mapping[str, str]) -> float:
    """Seconds between ingestion cycles (default 5.0)."""
    try:
        return max(0.1, float(env_map.get("RBTA_LIVE_POLL_INTERVAL_SEC", "5.0")))
    except (TypeError, ValueError):
        return 5.0


class LiveWorker:
    """Runs LiveIngestionCoordinator cycles on a background thread.

    A failed cycle is recorded (last_error, consecutive_failures) and the
    worker continues: the coordinator only commits transport cursors on
    success, so a failed cycle never advances state.
    """

    def __init__(
        self,
        service: LiveRBTAService,
        coordinator: Any,
        poll_interval: float = 5.0,
        drain_on_stop: bool = False,
        sleep_fn: Optional[Callable[[float], None]] = None,
        model_override: Optional[Union[bool, str]] = None,
        derivation_override: Optional[Union[bool, str]] = None,
    ) -> None:
        self.service = service
        self.coordinator = coordinator
        self.poll_interval = max(0.01, float(poll_interval))
        self.drain_on_stop = drain_on_stop
        self._sleep_fn = sleep_fn
        # Raw one-time override values forwarded from the bootstrap env_map
        # (None -> fall back to os.environ at pin time, so callers that only
        # set the process environment keep working).
        self.model_override = model_override
        self.derivation_override = derivation_override
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._guard = threading.Lock()
        self._state_lock = threading.Lock()
        self.cycles_completed = 0
        self.consecutive_failures = 0
        self.last_error: Optional[str] = None
        self.last_cycle_at: Optional[str] = None

    def start(self) -> bool:
        """Start the worker thread. Returns False if already running."""
        with self._guard:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._pin_model_version()
            self._thread = threading.Thread(
                target=self._run, name="rbta-live-worker", daemon=True
            )
            self._thread.start()
            return True

    def stop(self, drain: Optional[bool] = None) -> None:
        """Stop the worker thread (idempotent) and optionally drain buckets.

        Drain is cooperative: it only runs once the worker thread has exited.
        If the thread is still alive after the join timeout, the drain is
        skipped with a warning so an in-flight cycle is never raced.
        """
        with self._guard:
            self._stop_event.set()
            thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=_JOIN_TIMEOUT_SEC)
            if thread.is_alive():
                logger.warning(
                    "Live worker thread still alive after %.1fs join; "
                    "skipping drain to avoid racing an active cycle",
                    _JOIN_TIMEOUT_SEC,
                )
                return
        if drain if drain is not None else self.drain_on_stop:
            self.service.shutdown(drain=True)

    def is_alive(self) -> bool:
        """Whether the worker thread is currently running."""
        with self._guard:
            return self._thread is not None and self._thread.is_alive()

    def status(self) -> Dict[str, Any]:
        """Operational snapshot for the read-only live status API (L3)."""
        with self._state_lock:
            return {
                "alive": self.is_alive(),
                "cycles_completed": self.cycles_completed,
                "consecutive_failures": self.consecutive_failures,
                "last_error": self.last_error,
                "last_cycle_at": self.last_cycle_at,
            }

    def _pin_model_version(self) -> None:
        """Record the frozen scoring model version for this live session.

        A previously pinned version that differs from the current pipeline
        rejects startup (stale-model fail-fast), unless the operator passes
        a one-time override whose value equals the current version
        (``RBTA_LIVE_MODEL_OVERRIDE=<version>``) or the service carries a
        new ``run_id``. Missing ``model_version`` metadata is a hard error:
        the worker never falls back to ``"unknown"``.
        """
        pipeline = getattr(self.service, "scoring_pipeline", None)
        metadata = getattr(pipeline, "metadata", None)
        version = metadata.get("model_version") if isinstance(metadata, dict) else None
        if not version:
            raise RuntimeError(
                "Live worker requires scoring pipeline metadata with a non-empty "
                "'model_version'; refusing to start without a pinned model version."
            )
        version = str(version)
        source_state = self.service.get_live_source_state()
        old_pin = source_state.get("live_model_version")
        current_run = getattr(self.service, "run_id", None)
        old_run = source_state.get("live_run_id")
        self._reject_on_mismatch(
            kind="model",
            pinned=old_pin,
            current=version,
            override_raw=self.model_override,
            env_name="RBTA_LIVE_MODEL_OVERRIDE",
            current_run=current_run,
            old_run=old_run,
        )
        repinned_derivation = self._check_derivation_hash(source_state, current_run, old_run)
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "live_model_version": version,
            "live_worker_started_at": now,
        }
        # M1: first-seen start marker, written exactly once per state file.
        # Unlike live_worker_started_at (refreshed every start), this key is
        # only set when absent so restarts never overwrite the original.
        if "live_first_started_at" not in source_state:
            payload["live_first_started_at"] = now
        if current_run:
            payload["live_run_id"] = current_run
        prior = [
            e
            for e in (source_state.get("live_pin_history") or [])
            if isinstance(e, dict)
        ]
        if not prior or prior[-1].get("version") != version:
            prior.append({"version": version, "at": now})
            payload["live_pin_history"] = prior[-20:]
        if repinned_derivation is not None:
            base = payload.get("live_pin_history", prior)
            base.append({"kind": "derivation", "value": repinned_derivation, "at": now})
            payload["live_pin_history"] = base[-20:]
        self.service.update_live_source_state(payload)
        logger.info("Live worker pinned scoring model version: '%s'", version)

    @staticmethod
    def _resolve_override(raw: Any, env_name: str) -> Optional[str]:
        """Raw override value: explicit constructor value wins, else os.environ."""
        value = raw if raw is not None else os.environ.get(env_name)
        if value is None or value is False:
            return None
        text = str(value).strip()
        return text or None

    def _reject_on_mismatch(
        self,
        *,
        kind: str,
        pinned: Any,
        current: str,
        override_raw: Any,
        env_name: str,
        current_run: Any,
        old_run: Any,
    ) -> bool:
        """Fail fast on a stale pin; exact-value override or new run allows.

        The override is one-time and version-bound: it must equal the
        current target value, never a blanket ``"true"``. A set-but-wrong
        override raises instead of being ignored (fail-closed). Returns
        True when startup may continue.
        """
        if not pinned or pinned == current:
            return True
        new_run = bool(current_run) and current_run != old_run
        override = self._resolve_override(override_raw, env_name)
        if override is not None and override == current:
            logger.warning(
                "Live %s override accepted for target '%s' via %s; re-pinning.",
                kind,
                current,
                env_name,
            )
            return True
        if new_run:
            return True
        label = (
            f"pinned '{pinned}' in source state vs current '{current}'"
            if kind == "model"
            else f"stored '{pinned}' in source state vs current '{current}'"
        )
        noun = "model version" if kind == "model" else "derivation config"
        hint = f"set {env_name}='{current}' to override explicitly"
        if override is not None:
            raise RuntimeError(
                f"Live {noun} mismatch: {label}. Refusing to start; "
                f"provided {env_name} value did not match '{current}'. "
                f"To override, {hint} or restart with a new run_id."
            )
        raise RuntimeError(
            f"Live {noun} mismatch: {label}. Refusing to start; "
            f"{hint} or restart with a new run_id."
        )

    def _check_derivation_hash(
        self, source_state: Dict[str, Any], current_run: Any, old_run: Any
    ) -> Optional[str]:
        """Reject startup on derivation-config drift; store baseline when unset.

        Fail-closed: a hash that cannot be computed raises here and is never
        swallowed by the worker.

        Returns the re-pinned hash when a stored-vs-current mismatch was
        tolerated (exact-value override or new run_id) and the stored
        baseline was rewritten, else None. The caller audits a non-None
        return in ``live_pin_history`` so the next restart without an
        override succeeds against the new baseline.
        """
        state_manager = self.service.state_manager
        stored = state_manager.get_derivation_hash()
        current = compute_derivation_hash()
        self._reject_on_mismatch(
            kind="derivation",
            pinned=stored,
            current=current,
            override_raw=self.derivation_override,
            env_name="RBTA_LIVE_DERIVATION_OVERRIDE",
            current_run=current_run,
            old_run=old_run,
        )
        if not stored:
            state_manager.set_derivation_hash(current)
            return None
        if stored != current:
            # Mismatch was tolerated above (override accepted or new run):
            # re-pin the baseline so restarts without the override succeed.
            state_manager.set_derivation_hash(current)
            return current
        return None

    def _backoff_delay(self) -> float:
        """Exponential backoff with jitter after consecutive cycle failures.

        Grows from the base poll interval (``base * 2**(failures-1)``) up to
        a 5-minute cap; a small proportional jitter avoids lock-step retries.
        """
        base = max(0.01, float(self.poll_interval))
        failures = max(1, self.consecutive_failures)
        delay = min(base * (2.0 ** (failures - 1)), _BACKOFF_MAX_SEC)
        jitter = random.uniform(0.0, min(1.0, delay * 0.1))
        return delay + jitter

    def _sleep(self, delay: float) -> None:
        """Wait between cycles on the stop event (or the injected sleep_fn verbatim)."""
        (self._sleep_fn or self._stop_event.wait)(delay)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.coordinator.run_cycle()
            except Exception as exc:
                logger.error("Live ingestion cycle failed: %s", exc)
                with self._state_lock:
                    self.consecutive_failures += 1
                    self.last_error = str(exc)
                    delay = self._backoff_delay()
                self._sleep(delay)
                continue
            with self._state_lock:
                self.cycles_completed += 1
                self.consecutive_failures = 0
                self.last_cycle_at = datetime.now(timezone.utc).isoformat()
            self._sleep(self.poll_interval)
