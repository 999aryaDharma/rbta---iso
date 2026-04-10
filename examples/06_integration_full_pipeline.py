"""
INTEGRATION EXAMPLE 2: Stream → RBTA → Isolation Forest (FULL PIPELINE)
════════════════════════════════════════════════════════════════════════

Real-time simulation with complete anomaly detection pipeline.
This is closest to production setup.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming import AlertStreamSimulator
from src.engine.rbta_algorithm_02 import run_rbta
from src.engine.isolation_forest import run_pipeline as run_isolation_forest


class FullPipelineProcessor:
    """Stream → RBTA → IF scoring."""
    
    def __init__(self):
        self.raw_alerts = []
        self.processed_count = 0
        self.bucket_count = 0
        self.alert_count = 0
    
    def process_alert(self, alert):
        """Receive alert from simulator."""
        # Just buffer raw alerts (RBTA will handle bundling)
        self.raw_alerts.append({
            'timestamp_utc': alert.get('timestamp_utc'),
            'agent_id': alert.get('agent_id'),
            'agent_name': alert.get('agent_name'),
            'rule_id': alert.get('rule_id'),
            'rule_group_primary': alert.get('rule_group_primary'),
            'rule_level': alert.get('rule_level'),
            'srcip': alert.get('srcip'),
            'mitre_tactic': alert.get('mitre_tactic'),
            'payload': alert.get('payload'),
        })
        self.alert_count += 1


def main():
    print("=" * 70)
    print("INTEGRATION 2: Stream → RBTA → Isolation Forest")
    print("=" * 70)
    
    # Step 1: Setup simulator
    # speed_factor=100 means: real 100 seconds = 1 second simulation
    # Dataset spans ~24 minutes → takes ~14 seconds to stream
    sim = AlertStreamSimulator(
        csv_path="data/raw/rbta_ready_all.csv",
        speed_factor=100,           # 100x acceleration (real time / 100)
        log_file="output/06_stream_rbta_if.log",
        log_level="INFO",            # Show progress logs
        log_frequency=10000,         # Progress every 10k alerts
    )
    
    processor = FullPipelineProcessor()
    
    # Step 2: STREAMING PHASE
    print("\n[STEP 1] Streaming alerts (with time-accurate replay)...")
    print("  ├─ Data spans ~24 minutes of real time")
    print("  ├─ Speed factor: 100x (real time acceleration)")
    print("  ├─ Expected duration: ~14-16 seconds")
    print("  └─ Timestamps preserved (each alert respects original arrival time)")
    print()
    
    sim.on_alert(processor.process_alert)
    start = datetime.now()
    stats = sim.replay()
    duration_stream = (datetime.now() - start).total_seconds()
    
    print(f"\n✓ Streamed {len(processor.raw_alerts):,} alerts in {duration_stream:.1f} sec")
    print(f"  Rate: {len(processor.raw_alerts)/duration_stream:.0f} alerts/sec")
    print(f"  Real time span: {stats.duration_sec:.1f} sec (at 100x speed)")
    
    # Step 3: RBTA BUCKETING
    print(f"\n[STEP 2] RBTA bucketing...")
    
    # Convert to DataFrame
    df_alerts = pd.DataFrame(processor.raw_alerts)
    df_alerts = df_alerts.rename(columns={
        'timestamp_utc': 'timestamp',
        'rule_group_primary': 'rule_groups',
        'rule_level': 'rule_level',
    })
    
    start = datetime.now()
    df_meta, df_compound, idx_map, elastic_win, wmark = run_rbta(
        df=df_alerts,
        delta_t_minutes=5,
        max_window_minutes=60,
        buffer_size=50,
    )
    duration_rbta = (datetime.now() - start).total_seconds()
    
    print(f"✓ Produced {len(df_meta):,} buckets in {duration_rbta:.1f} sec")
    if len(df_meta) > 0:
        print(f"  Compression: {len(processor.raw_alerts)/len(df_meta):.1f}x")
    
    # Step 4: ISOLATION FOREST SCORING
    print(f"\n[STEP 3] Isolation Forest scoring {len(df_meta):,} buckets...")
    
    # Save RBTA buckets to temporary CSV for IF pipeline
    temp_csv = Path("output/temp_rbta_for_if.csv")
    temp_csv.parent.mkdir(exist_ok=True)
    df_meta.to_csv(temp_csv, index=False)
    
    scored_buckets = None
    try:
        # Call run_pipeline function
        df_scored, model, scaler, theta = run_isolation_forest(
            csv_path=str(temp_csv),
            output_dir="output",
            contamination=0.05,
            n_estimators=200,
            theta_method="iqr",
            theta_override=None,
            random_state=42,
        )
        scored_buckets = df_scored
        
    except Exception as e:
        print(f"⚠ Warning: IF scoring failed ({e})")
        print("  → Model may not be trained yet")
        print("  → Run: python main.py  (to train model first)")
        print("  → Skipping IF scoring phase")
        return
    
    duration_if = (datetime.now() - start).total_seconds()
    
    print(f"✓ Scored {len(scored_buckets):,} buckets in {duration_if:.1f} sec")
    
    # Step 5: ANALYSIS
    print(f"\n[STEP 4] Pipeline Results:")
    
    # Severity breakdown
    sev_dist = {}
    for _, bucket in scored_buckets.iterrows():
        sev = int(bucket.get('rule_level', 0))
        sev_dist[sev] = sev_dist.get(sev, 0) + 1
    
    print(f"\n  Severity Distribution:")
    for sev in sorted(sev_dist.keys()):
        print(f"    L{sev}: {sev_dist[sev]:,}")
    
    # Anomaly counts
    anomalies = scored_buckets[scored_buckets.get('escalate') == 1]
    print(f"\n  Anomalies detected: {len(anomalies):,} ({100*len(anomalies)/len(scored_buckets):.1f}%)")
    
    # Agent breakdown
    agent_anom = {}
    for _, bucket in anomalies.iterrows():
        agent = str(bucket.get('agent_name', 'unknown'))
        agent_anom[agent] = agent_anom.get(agent, 0) + 1
    
    if agent_anom:
        print(f"\n  Top anomalous agents:")
        for agent, count in sorted(agent_anom.items(), key=lambda x: -x[1])[:5]:
            print(f"    {agent}: {count}")
    
    # Timing summary
    total_time = duration_stream + duration_rbta + duration_if
    print(f"\n[STEP 5] Timing Summary:")
    print(f"  Streaming: {duration_stream:.1f}s ({100*duration_stream/total_time:.0f}%)")
    print(f"  RBTA:      {duration_rbta:.1f}s ({100*duration_rbta/total_time:.0f}%)")
    print(f"  IF:        {duration_if:.1f}s ({100*duration_if/total_time:.0f}%)")
    print(f"  TOTAL:     {total_time:.1f}s")
    
    # Export sample anomalies
    print(f"\n[STEP 6] Exporting results...")
    output_file = Path("output/06_stream_pipeline_anomalies.json")
    
    anomaly_export = []
    for _, bucket in anomalies.head(100).iterrows():  # First 100
        anomaly_export.append({
            'agent_id': str(bucket.get('agent_id', '')),
            'agent_name': str(bucket.get('agent_name', '')),
            'rule_group': str(bucket.get('rule_groups', '')),
            'severity': int(bucket.get('rule_level', 0)),
            'alert_count': int(bucket.get('alert_count', 1)),
            'anomaly_score': float(bucket.get('anomaly_score', 0.0)),
            'decision': str(bucket.get('decision', 'UNKNOWN')),
            'action': str(bucket.get('action', 'SUPPRESS')),
        })
    
    with open(output_file, 'w') as f:
        json.dump(anomaly_export, f, indent=2, default=str)
    
    print(f"✓ Exported {len(anomaly_export):,} anomalies to {output_file.name}")
    
    print("\n" + "=" * 70)
    print("✓ INTEGRATION 2 Complete (Full Pipeline)")
    print("=" * 70)


if __name__ == "__main__":
    main()
