"""Historical ingestion checkpoint persistence module."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, List, Optional, Union


@dataclass
class HistoricalCheckpoint:
    """State of historical ingestion progress across daily indices.

    Attributes
    ----------
    mode : str
        Ingestion mode ("historical").
    current_index : str | None
        Name of the daily index currently being processed.
    last_sort : list[Any] | None
        OpenSearch/Wazuh Indexer search_after cursor array.
    processed_count : int
        Total number of raw alerts processed/yielded.
    last_wazuh_alert_id : str | None
        Last successfully emitted Wazuh alert ID.
    completed_indices : list[str]
        List of daily index names fully exhausted.
    updated_at : str
        UTC ISO-8601 timestamp of the last checkpoint update.
    """

    mode: str = "historical"
    current_index: Optional[str] = None
    last_sort: Optional[List[Any]] = None
    processed_count: int = 0
    last_wazuh_alert_id: Optional[str] = None
    completed_indices: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update(
        self,
        index_name: str,
        last_sort: List[Any],
        wazuh_alert_id: str,
    ) -> None:
        """Update cursor position after processing an alert or page."""
        self.current_index = index_name
        self.last_sort = list(last_sort)
        self.last_wazuh_alert_id = wazuh_alert_id
        self.processed_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_index_completed(self, index_name: str) -> None:
        """Mark a daily index as completed and reset the active cursor."""
        if index_name not in self.completed_indices:
            self.completed_indices.append(index_name)
        self.current_index = None
        self.last_sort = None
        self.updated_at = datetime.now(timezone.utc).isoformat()


class CheckpointManager:
    """Manages disk serialization and restoration of HistoricalCheckpoint objects."""

    def __init__(self, filepath: Union[str, Path] = "checkpoints/historical_checkpoint.json") -> None:
        self.filepath: Path = Path(filepath).resolve()

    def load(self) -> HistoricalCheckpoint:
        """Load checkpoint from JSON file, or return fresh checkpoint if file does not exist."""
        if not self.filepath.exists():
            return HistoricalCheckpoint()

        try:
            with self.filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return HistoricalCheckpoint(
                mode=str(data.get("mode", "historical")),
                current_index=data.get("current_index"),
                last_sort=data.get("last_sort"),
                processed_count=int(data.get("processed_count", 0)),
                last_wazuh_alert_id=data.get("last_wazuh_alert_id"),
                completed_indices=list(data.get("completed_indices", [])),
                updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
            )
        except Exception:
            return HistoricalCheckpoint()

    def save(self, checkpoint: HistoricalCheckpoint) -> None:
        """Save checkpoint to disk atomically via temporary file."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.filepath.with_suffix(".tmp")

        data = asdict(checkpoint)
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        tmp_file.replace(self.filepath)
