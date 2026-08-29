"""Isolation Forest scoring pipeline and reference training module."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import sklearn

from src.contracts.meta_alert import MetaAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.features.extractor import FEATURE_COLUMNS, SevenFeatureExtractor
from src.model.calibration import ScoreCalibration
from src.model.decision import evaluate_decision
from src.model.threshold import TukeyThreshold, compute_tukey_threshold


@dataclass(frozen=True)
class ModelArtifactBundle:
    """In-memory bundle of all 6 components constituting a trained model artifact."""

    scaler: RobustScaler
    model: IsolationForest
    calibration: ScoreCalibration
    threshold: TukeyThreshold
    metadata: Dict[str, Any]
    schema: Dict[str, Any]


def train_reference_pipeline(
    metas: Sequence[MetaAlert],
    random_state: int = 42,
    model_version: str = "rbta-if-v1",
    training_run_id: Optional[str] = None,
    git_commit: Optional[str] = None,
    research_config_hash: Optional[str] = None,
) -> ModelArtifactBundle:
    """Train reference Isolation Forest model and generate calibrated artifact bundle.

    Parameters
    ----------
    metas : Sequence[MetaAlert]
        Training reference meta-alerts.
    random_state : int
        Fixed random seed for reproducibility.
    model_version : str
        Version identifier for the trained model.
    training_run_id : str | None
        Unique identifier for the training run.
    git_commit : str | None
        Git commit SHA-256 for research reproducibility.
    research_config_hash : str | None
        Cryptographic hash of the research hyperparameters.

    Returns
    -------
    ModelArtifactBundle
        Complete bundle ready for serialization and inference.
    """
    if len(metas) < 4:
        raise ValueError(f"At least 4 meta-alerts are required to train reference pipeline, got {len(metas)}")

    # 1. Extract 7 canonical features as DataFrame with feature names
    X_df = SevenFeatureExtractor.extract_features_df(metas)

    # 2. Fit RobustScaler
    scaler = RobustScaler()
    scaler.fit(X_df)
    X_scaled = scaler.transform(X_df)

    # 3. Fit IsolationForest (200 trees, contamination="auto")
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # 4. Derive Raw Anomaly Scores (higher = more anomalous)
    raw_scores = -model.score_samples(X_scaled)
    raw_min = float(np.min(raw_scores))
    raw_max = float(np.max(raw_scores))

    # 5. Fit Score Calibration
    calibration = ScoreCalibration(
        raw_min=raw_min,
        raw_max=raw_max,
        higher_is_more_anomalous=True,
        version="minmax-v1",
    )

    # 6. Derive Calibrated Scores and Tukey IQR Threshold
    cal_scores = [calibration.calibrate(float(s)) for s in raw_scores]
    threshold = compute_tukey_threshold(cal_scores)

    # 7. Metadata and Schema
    schema = {
        "schema_version": "1.0",
        "features": list(FEATURE_COLUMNS),
    }

    start_times = [m.start_time for m in metas if m.start_time is not None]
    end_times = [m.end_time for m in metas if m.end_time is not None]

    if git_commit is not None:
        resolved_git_commit = git_commit
    else:
        try:
            resolved_git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            resolved_git_commit = "unavailable"

    resolved_run_id = training_run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if research_config_hash is not None:
        resolved_config_hash = research_config_hash
    else:
        config_str = f"model_version={model_version},random_state={random_state},contamination=auto,n_estimators=200"
        resolved_config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    metadata = {
        "model_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run_id": resolved_run_id,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "n_estimators": 200,
        "contamination": "auto",
        "random_state": random_state,
        "training_row_count": len(metas),
        "meta_alert_count": len(metas),
        "training_period_start": min(start_times).isoformat() if start_times else None,
        "training_period_end": max(end_times).isoformat() if end_times else None,
        "score_calibration_version": calibration.version,
        "git_commit": resolved_git_commit,
        "research_config_hash": resolved_config_hash,
        "feature_schema_version": schema["schema_version"],
    }

    return ModelArtifactBundle(
        scaler=scaler,
        model=model,
        calibration=calibration,
        threshold=threshold,
        metadata=metadata,
        schema=schema,
    )


class ScoringPipeline:
    """Inference scoring pipeline executing strictly in read-only prediction mode."""

    def __init__(self, bundle: ModelArtifactBundle) -> None:
        self.scaler: RobustScaler = bundle.scaler
        self.model: IsolationForest = bundle.model
        self.calibration: ScoreCalibration = bundle.calibration
        self.threshold: TukeyThreshold = bundle.threshold
        self.metadata: Dict[str, Any] = bundle.metadata
        self.schema: Dict[str, Any] = bundle.schema

        # Verify schema matches canonical FEATURE_COLUMNS
        schema_features = tuple(self.schema.get("features", []))
        if schema_features != FEATURE_COLUMNS:
            raise ValueError(f"Bundle feature schema {schema_features} != expected {FEATURE_COLUMNS}")

    def score_single(self, meta: MetaAlert) -> ScoredMetaAlert:
        """Score a single MetaAlert event in stream/live mode without fitting.

        Parameters
        ----------
        meta : MetaAlert
            Incoming MetaAlert DTO.

        Returns
        -------
        ScoredMetaAlert
            Immutable scored result DTO.
        """
        features_dict = SevenFeatureExtractor.extract_features_dict(meta)
        X_single_df = pd.DataFrame([features_dict], columns=list(FEATURE_COLUMNS))

        # Use persisted scaler to transform single row
        X_scaled = self.scaler.transform(X_single_df)

        # Inference only: score_samples
        raw_score = float(-self.model.score_samples(X_scaled)[0])
        anomaly_score = float(self.calibration.calibrate(raw_score))

        decision, action, escalate = evaluate_decision(
            anomaly_score=anomaly_score,
            threshold=self.threshold.threshold,
            max_severity=meta.max_severity,
            alert_count=meta.alert_count,
            mitre_tactic_count=len(meta.mitre_tactics_unique),
        )

        return ScoredMetaAlert(
            meta_id=meta.meta_id,
            agent_id=meta.agent_id,
            agent_name=meta.agent_name,
            rule_group_primary=meta.rule_group_primary,
            start_time=meta.start_time,
            end_time=meta.end_time,
            alert_count=meta.alert_count,
            max_severity=meta.max_severity,
            mitre_tactics=meta.mitre_tactics_unique,
            seven_features=features_dict,
            raw_model_score=raw_score,
            anomaly_score=anomaly_score,
            threshold_used=self.threshold.threshold,
            decision=decision,
            action=action,
            escalate=escalate,
            model_version=str(self.metadata.get("model_version", "unknown")),
            feature_schema_version=str(self.schema.get("schema_version", "1.0")),
            score_calibration_version=str(self.calibration.version),
            source_alert_ids=meta.wazuh_alert_ids,
        )

    def score_meta_alerts(
        self,
        metas: Sequence[MetaAlert],
    ) -> Tuple[pd.DataFrame, List[ScoredMetaAlert]]:
        """Score a sequence of MetaAlerts in batch mode.

        Parameters
        ----------
        metas : Sequence[MetaAlert]
            Sequence of MetaAlert DTOs.

        Returns
        -------
        Tuple[pd.DataFrame, List[ScoredMetaAlert]]
            DataFrame of scored results and corresponding list of ScoredMetaAlert objects.
        """
        scored_list = [self.score_single(m) for m in metas]
        rows = [
            {
                "meta_id": s.meta_id,
                "agent_id": s.agent_id,
                "agent_name": s.agent_name,
                "rule_group_primary": s.rule_group_primary,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "alert_count": s.alert_count,
                "max_severity": s.max_severity,
                "raw_model_score": s.raw_model_score,
                "anomaly_score": s.anomaly_score,
                "threshold_used": s.threshold_used,
                "decision": s.decision,
                "action": s.action,
                "escalate": s.escalate,
                "model_version": s.model_version,
                **dict(s.seven_features),
            }
            for s in scored_list
        ]
        df_scored = pd.DataFrame(rows)
        return df_scored, scored_list
