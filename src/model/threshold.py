"""Tukey IQR anomaly threshold calculation module."""

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import numpy as np


class ThresholdError(ValueError):
    """Raised when threshold cannot be computed from score distribution."""
    pass


@dataclass(frozen=True)
class TukeyThreshold:
    """Tukey IQR anomaly threshold definition.

    Attributes
    ----------
    q1 : float
        25th percentile of normalized reference scores.
    q3 : float
        75th percentile of normalized reference scores.
    iqr : float
        Interquartile range (q3 - q1).
    threshold : float
        Outlier fence threshold (q3 + 1.5 * iqr).
    method : str
        Methodology identifier.
    """

    q1: float
    q3: float
    iqr: float
    threshold: float
    method: str = "tukey_iqr"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize threshold to dictionary."""
        return {
            "method": self.method,
            "q1": float(self.q1),
            "q3": float(self.q3),
            "iqr": float(self.iqr),
            "threshold": float(self.threshold),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TukeyThreshold":
        """Deserialize threshold from dictionary."""
        return cls(
            q1=float(data["q1"]),
            q3=float(data["q3"]),
            iqr=float(data["iqr"]),
            threshold=float(data["threshold"]),
            method=str(data.get("method", "tukey_iqr")),
        )


def compute_tukey_threshold(scores: Sequence[float]) -> TukeyThreshold:
    """Compute Tukey IQR threshold on a sequence of reference anomaly scores.

    Formula:
        Q1 = percentile 25
        Q3 = percentile 75
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR

    Note: The threshold is NOT clamped to <= 1.0.

    Parameters
    ----------
    scores : Sequence[float]
        Reference normalized anomaly scores.

    Returns
    -------
    TukeyThreshold
        Calculated threshold dataclass.

    Raises
    ------
    ThresholdError
        If fewer than 4 score samples are provided.
    """
    if len(scores) < 4:
        raise ThresholdError(f"At least 4 score samples are required to compute Tukey IQR, got {len(scores)}")

    arr = np.asarray(scores, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    threshold = q3 + (1.5 * iqr)

    return TukeyThreshold(q1=q1, q3=q3, iqr=iqr, threshold=threshold)
