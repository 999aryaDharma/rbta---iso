"""Model anomaly score calibration module."""

from dataclasses import dataclass
from typing import Any, Dict


class CalibrationError(ValueError):
    """Raised when score calibration bounds are invalid or degenerate."""
    pass


@dataclass(frozen=True)
class ScoreCalibration:
    """Stream-safe min-max score calibration learned during reference/training runs.

    Attributes
    ----------
    raw_min : float
        Minimum raw anomaly score observed on reference dataset.
    raw_max : float
        Maximum raw anomaly score observed on reference dataset.
    higher_is_more_anomalous : bool
        Whether higher raw score corresponds to greater anomaly (default True).
    version : str
        Calibration policy identifier.
    """

    raw_min: float
    raw_max: float
    higher_is_more_anomalous: bool = True
    version: str = "minmax-v1"

    def __post_init__(self) -> None:
        if self.raw_max <= self.raw_min:
            raise CalibrationError(
                f"Degenerate calibration: raw_max must be strictly greater than raw_min, got raw_min={self.raw_min}, raw_max={self.raw_max}"
            )

    def calibrate(self, raw_score: float) -> float:
        """Calibrate a raw anomaly score into a reference-normalized anomaly score.

        Parameters
        ----------
        raw_score : float
            Raw model anomaly score (e.g. -score_samples).

        Returns
        -------
        float
            Normalized anomaly score where reference baseline is ~[0.0, 1.0].
        """
        return (raw_score - self.raw_min) / (self.raw_max - self.raw_min)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration parameters to dictionary."""
        return {
            "version": self.version,
            "raw_min": float(self.raw_min),
            "raw_max": float(self.raw_max),
            "higher_is_more_anomalous": bool(self.higher_is_more_anomalous),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreCalibration":
        """Deserialize calibration parameters from dictionary."""
        return cls(
            raw_min=float(data["raw_min"]),
            raw_max=float(data["raw_max"]),
            higher_is_more_anomalous=bool(data.get("higher_is_more_anomalous", True)),
            version=str(data.get("version", "minmax-v1")),
        )
