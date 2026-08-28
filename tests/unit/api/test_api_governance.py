"""Governance tests for API package (Sprint 9)."""
from pathlib import Path

API_SRC = Path(__file__).resolve().parent.parent.parent.parent / "src" / "api"


def test_no_research_logic_in_api_adapters():
    """Verify that Shuffle adapter and Telegram formatter contain zero algorithmic/model logic."""
    forbidden = [
        "IsolationForest",
        "RobustScaler",
        "AgentTemporalState",
        "calculate_baseline",
        "compute_tukey_threshold",
    ]

    for filename in ["shuffle_adapter.py", "telegram_formatter.py"]:
        file_path = API_SRC / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            for f in forbidden:
                assert f not in content, f"Forbidden logic symbol '{f}' found in {filename}"
