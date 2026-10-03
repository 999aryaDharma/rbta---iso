"""L3: read-only live status API contract (TDD RED)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from starlette.testclient import TestClient

from src.api.app import create_app
from src.runtime.durable_state import DurableStateManager
from src.runtime.service import LiveRBTAService


API_KEY = "live-status-test-key"


def _make_service(tmp_path, source_state=None):
    state_mgr = DurableStateManager(tmp_path / "state.json")
    service = LiveRBTAService(
        scoring_pipeline=MagicMock(),
        state_manager=state_mgr,
        auto_persist=False,
    )
    if source_state:
        service.source_checkpoint.update(source_state)
    return service


def _make_worker(status_dict):
    worker = SimpleNamespace()
    worker.status = lambda: dict(status_dict)
    return worker


def _client(service, worker, api_key=API_KEY):
    app = create_app(service=service, api_key=api_key)
    app.state.live_worker = worker
    return TestClient(app)


def _auth():
    return {"Authorization": f"Bearer {API_KEY}"}


def test_live_status_contract_shape(tmp_path):
    """GET /api/v1/live/status returns 200 with the locked L3 response shape."""
    cursor = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    service = _make_service(
        tmp_path,
        {"recent_poll_cursor": cursor, "live_model_version": "v-test-1"},
    )
    worker = _make_worker(
        {
            "alive": True,
            "cycles_completed": 7,
            "consecutive_failures": 0,
            "last_error": None,
            "last_cycle_at": cursor,
        }
    )
    client = _client(service, worker)
    resp = client.get("/api/v1/live/status", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "worker_alive",
        "cycles_completed",
        "consecutive_failures",
        "last_error",
        "last_cycle_at",
        "live_model_version",
        "recent_poll_cursor",
        "lag_sec",
        "buffer_size",
        "outbox_pending",
    ):
        assert key in data, f"missing key: {key}"
    assert data["worker_alive"] is True
    assert data["cycles_completed"] == 7
    assert data["consecutive_failures"] == 0
    assert data["last_error"] is None
    assert data["live_model_version"] == "v-test-1"
    assert data["recent_poll_cursor"] == cursor
    assert data["lag_sec"] is not None and data["lag_sec"] >= 0
    assert data["buffer_size"] is None  # F11: disabled/absent buffer is unmeasured, not empty
    assert data["outbox_pending"] == 0


def test_live_status_worker_none_null_safe(tmp_path):
    """Worker disabled (None) still returns 200 with null-safe worker fields."""
    service = _make_service(tmp_path, {})
    client = _client(service, None)
    resp = client.get("/api/v1/live/status", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["worker_alive"] is False
    assert data["cycles_completed"] == 0
    assert data["consecutive_failures"] == 0
    assert data["last_error"] is None
    assert data["last_cycle_at"] is None
    assert data["buffer_size"] is None  # F11: disabled/absent buffer is unmeasured, not empty
    assert data["outbox_pending"] == 0


def test_live_status_read_only(tmp_path):
    """Calling the endpoint repeatedly must not mutate service state."""
    cursor = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    service = _make_service(
        tmp_path,
        {"recent_poll_cursor": cursor, "live_model_version": "v-test-1"},
    )
    worker = _make_worker(
        {
            "alive": True,
            "cycles_completed": 3,
            "consecutive_failures": 0,
            "last_error": None,
            "last_cycle_at": cursor,
        }
    )
    client = _client(service, worker)
    before_checkpoint = dict(service.source_checkpoint)
    before_history = len(service.finalized_history)
    for _ in range(5):
        resp = client.get("/api/v1/live/status", headers=_auth())
        assert resp.status_code == 200
    assert dict(service.source_checkpoint) == before_checkpoint
    assert len(service.finalized_history) == before_history


def test_live_status_lag_null_when_cursor_empty(tmp_path):
    """lag_sec is null when there is no poll cursor yet."""
    service = _make_service(tmp_path, {})
    worker = _make_worker(
        {
            "alive": False,
            "cycles_completed": 0,
            "consecutive_failures": 0,
            "last_error": None,
            "last_cycle_at": None,
        }
    )
    client = _client(service, worker)
    resp = client.get("/api/v1/live/status", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["recent_poll_cursor"] is None
    assert data["lag_sec"] is None


def test_live_status_auth_rejected_without_key(tmp_path):
    """Requests without a valid key are rejected like other endpoints."""
    service = _make_service(tmp_path, {})
    worker = _make_worker(
        {
            "alive": False,
            "cycles_completed": 0,
            "consecutive_failures": 0,
            "last_error": None,
            "last_cycle_at": None,
        }
    )
    client = _client(service, worker)
    resp = client.get("/api/v1/live/status")
    assert resp.status_code == 401


# --- F11 honest telemetry (TDD) ---

def _make_buffer_worker(status_dict, held_size):
    """Worker whose coordinator exposes an enabled OrderBuffer of held_size."""
    buf = MagicMock()
    buf.status.return_value = {"size": held_size}
    coord = SimpleNamespace(order_buffer=buf)
    worker = SimpleNamespace(coordinator=coord)
    worker.status = lambda: dict(status_dict)
    return worker


def test_live_status_buffer_size_none_when_disabled(tmp_path):
    """F11/F5: absent buffer reports None (unmeasured), never fake 0."""
    service = _make_service(tmp_path, {})
    worker = _make_worker(
        {"alive": False, "cycles_completed": 0, "consecutive_failures": 0,
         "last_error": None, "last_cycle_at": None}
    )
    assert getattr(worker, "coordinator", None) is None
    client = _client(service, worker)
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["buffer_size"] is None


def test_live_status_buffer_size_reports_held_count_when_enabled(tmp_path):
    """F11: enabled buffer reports real held size via worker.coordinator.order_buffer."""
    service = _make_service(tmp_path, {})
    worker = _make_buffer_worker(
        {"alive": True, "cycles_completed": 1, "consecutive_failures": 0,
         "last_error": None, "last_cycle_at": None},
        held_size=4,
    )
    client = _client(service, worker)
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["buffer_size"] == 4


def test_live_status_dispatcher_null_when_absent(tmp_path):
    """F11: no dispatcher wired -> dispatcher block is null (guarded getattr)."""
    service = _make_service(tmp_path, {})
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert "dispatcher" in data
    assert data["dispatcher"] is None


def test_live_status_dispatcher_status_passthrough(tmp_path):
    """F11: wired dispatcher with .status() is passed through; errors -> null."""
    service = _make_service(tmp_path, {})
    client = _client(service, _make_worker({"alive": False}))
    client.app.state.telegram_dispatcher = SimpleNamespace(
        status=lambda: {"enabled": True, "sent": 2}
    )
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["dispatcher"] == {"enabled": True, "sent": 2}

    client.app.state.telegram_dispatcher = SimpleNamespace(
        status=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["dispatcher"] is None


def test_live_status_quarantine_null_when_unsupported(tmp_path, monkeypatch):
    """F11: state_manager without quarantine_list -> quarantine_total null."""
    service = _make_service(tmp_path, {})
    monkeypatch.delattr(type(service.state_manager), "quarantine_list", raising=False)
    monkeypatch.delattr(type(service.state_manager), "quarantine_count", raising=False)
    assert not hasattr(service.state_manager, "quarantine_list")
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert "quarantine_total" in data
    assert data["quarantine_total"] is None


def test_live_status_quarantine_total_when_supported(tmp_path):
    """F11: state_manager with quarantine_list -> real count."""
    service = _make_service(tmp_path, {})
    service.state_manager.quarantine_count = lambda: 2
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["quarantine_total"] == 2


def test_live_status_event_lag_null_when_no_scored_events(tmp_path):
    """F11: empty history+outbox -> newest_scored_event_time/event_lag_sec null."""
    service = _make_service(tmp_path, {})
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["newest_scored_event_time"] is None
    assert data["event_lag_sec"] is None


def test_live_status_event_lag_from_history_end_time(tmp_path):
    """F11: newest end_time across history/outbox drives event_lag_sec."""
    from types import SimpleNamespace as SN

    now = datetime.now(timezone.utc)
    service = _make_service(tmp_path, {})
    service.get_history = lambda: [
        SN(end_time=now - timedelta(minutes=10)),
        SN(end_time=now - timedelta(minutes=2)),
    ]
    service.get_outbox = lambda: [SN(end_time=now - timedelta(minutes=5))]
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["newest_scored_event_time"] is not None
    parsed = datetime.fromisoformat(data["newest_scored_event_time"])
    assert abs((parsed - (now - timedelta(minutes=2))).total_seconds()) < 5
    assert data["event_lag_sec"] is not None
    assert 100 <= data["event_lag_sec"] <= 180


def test_live_status_last_error_redacts_host(tmp_path):
    """F11: host/credentials stripped from last_error (no scheme://host leak)."""
    service = _make_service(tmp_path, {})
    worker = _make_worker(
        {"alive": True, "cycles_completed": 1, "consecutive_failures": 1,
         "last_error": "Connection refused https://wazuh.internal:9200/wazuh-alerts/_search",
         "last_cycle_at": None}
    )
    client = _client(service, worker)
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["last_error"] is not None
    assert "wazuh.internal" not in data["last_error"]
    assert "[host]" in data["last_error"]


def test_live_status_quarantine_prefers_count_over_list(tmp_path):
    """Minor: quarantine_total prefers COUNT(*) helper over loading rows."""
    service = _make_service(tmp_path, {})
    service.state_manager.quarantine_count = lambda: 41
    service.state_manager.quarantine_list = lambda: (_ for _ in ()).throw(
        AssertionError("list must not be loaded when count exists")
    )
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["quarantine_total"] == 41


def test_live_status_buffer_stats_block(tmp_path):
    """Minor: full buffer counters exposed when the buffer is enabled."""
    service = _make_service(tmp_path, {})
    worker = _make_worker({"alive": True})
    worker.coordinator = SimpleNamespace(
        order_buffer=SimpleNamespace(
            status=lambda: {
                "size": 3,
                "late_total": 1,
                "future_anomalies": 0,
                "backpressure_count": 2,
            }
        )
    )
    client = _client(service, worker)
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["buffer_size"] == 3
    assert data["buffer_stats"] == {
        "size": 3,
        "late_total": 1,
        "future_anomalies": 0,
        "backpressure_count": 2,
    }


def test_live_status_buffer_stats_null_when_disabled(tmp_path):
    """Minor: buffer_stats null (not zeros) when the buffer is disabled."""
    service = _make_service(tmp_path, {})
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["buffer_size"] is None
    assert data["buffer_stats"] is None


def test_live_status_event_lag_from_ingested_time(tmp_path):
    """Minor: event_lag_sec measured from newest ingested event, not scored flush."""
    now = datetime.now(timezone.utc)
    service = _make_service(
        tmp_path,
        {"newest_ingested_event_time": (now - timedelta(seconds=45)).isoformat()},
    )
    client = _client(service, _make_worker({"alive": True}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["newest_ingested_event_time"] is not None
    assert 30 <= data["event_lag_sec"] <= 120


def test_live_status_tls_verify_passthrough(tmp_path):
    """Minor: tls_verify operator choice visible in status."""
    service = _make_service(tmp_path, {"tls_verify": False})
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["tls_verify"] is False


def test_live_status_ingested_total_counts_seen_ids(tmp_path):
    """ingested_total exposes durable unique-alert count (follows F11 honesty)."""
    service = _make_service(tmp_path, {})
    service.state_manager.count_seen_alert_ids = lambda: 1899
    client = _client(service, _make_worker({"alive": True}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["ingested_total"] == 1899


def test_live_status_ingested_total_null_when_unsupported(tmp_path, monkeypatch):
    """ingested_total null (not 0) when the counter is unavailable."""
    service = _make_service(tmp_path, {})
    monkeypatch.delattr(type(service.state_manager), "count_seen_alert_ids", raising=False)
    client = _client(service, _make_worker({"alive": False}))
    data = client.get("/api/v1/live/status", headers=_auth()).json()
    assert data["ingested_total"] is None
