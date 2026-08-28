"""Feature engineering package for RBTA + Isolation Forest pipeline."""

from src.features.extractor import (
    FEATURE_COLUMNS,
    FeatureExtractionError,
    SevenFeatureExtractor,
    compute_rule_diversity_shannon,
    compute_severity_dispersion,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FeatureExtractionError",
    "SevenFeatureExtractor",
    "compute_rule_diversity_shannon",
    "compute_severity_dispersion",
]
