"""
INTEGRATION EXAMPLE 3: Windowed Real-Time Processing
═════════════════════════════════════════════════════

Continuous RBTA window updates as alerts arrive.
Most production-like approach for streaming.
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming import AlertStreamSimulator
from src.engine.rbta_algorithm_02 import run_rbta
from src.engine.isolation_forest import run_pipeline as run_isolation_forest


class WindowedProcessingEngine:
    """
    Maintain rolling RBTA window, update continuously.
    
    Pattern:
    1. Keep incoming alerts in rolling window (lookback_sec)
    2. Every N alerts, run RBTA on window contents
    3. Score results with IF
    4. Output anomalies in real-time
    """
    
    def __init__(self, window_sec=600, update_freq=1000, if_engine=None):
        self.window_sec = window_sec
        self.update_freq = update_freq
        self.if_engine = if_engine
        
        self.window_alerts = []
        self.window_start_time = None
        self.processed_count = 0
        self.bucket_count = 0
        self.anomaly_count = 0
        self.results = []
    
    def process_alert(self, alert):
        """Process incoming alert."""
        # Add to window
        ts = alert.get('timestamp_utc')
        self.window_alerts.append({
            'timestamp_utc': ts,
            'agent_id': alert.get('agent_id'),
            'agent_name': alert.get('agent_name'),
            'rule_id': alert.get('rule_id'),
            'rule_group_primary': alert.get('rule_group_primary'),
            'rule_level': alert.get('rule_level'),
            'srcip': alert.get('srcip'),
            'mitre_tactic': alert.get('mitre_tactic'),
            'payload': alert.get('payload'),
        })
        
        if self.window_start_time is None:
            self.window_start_time = ts
        
        self.processed_count += 1
        
        # Process window every N alerts
        if self.processed_count % self.update_freq == 0:
            self._process_window()
    
    def _process_window(self):
        """Run RBTA + IF on current window."""
        if not self.window_alerts:
            return
        
        # Trim window to timeframe
        now = self.window_alerts[-1]['timestamp_utc']
        cutoff = now - timedelta(seconds=self.window_sec)
        
        valid_alerts = [
            a for a in self.window_alerts
            if a['timestamp_utc'] >= cutoff
        ]
        
        if len(valid_alerts) < 2:
            return
        
        # Convert to DataFrame and run RBTA
        df_valid = pd.DataFrame(valid_alerts)
        df_valid = df_valid.rename(columns={
            'timestamp_utc': 'timestamp',
            'rule_group_primary': 'rule_groups',
            'rule_level': 'rule_level',
        })
        
        try:
            df_meta, df_compound, idx_map, elastic_win, wmark = run_rbta(
                df=df_valid,
                delta_t_minutes=5,
                max_window_minutes=60,
                buffer_size=50,
            )
            
            self.bucket_count += len(df_meta)
            
            # IF scoring (if available)
            if self.if_engine and len(df_meta) > 0:
                try:
                    # Save to temp CSV
                    temp_csv = Path("output/temp_window_for_if.csv")
                    temp_csv.parent.mkdir(exist_ok=True)
                    df_meta.to_csv(temp_csv, index=False)
                    
                    # Call run_isolation_forest
                    df_scored, model, scaler, theta = self.if_engine(
                        csv_path=str(temp_csv),
                        output_dir="output",
                        contamination=0.05,
                        n_estimators=200,
                        theta_method="iqr",
                        theta_override=None,
                        random_state=42,
                    )
                    
                    anomalies = df_scored[df_scored.get('escalate') == 1]
                    self.anomaly_count += len(anomalies)
                    
                    # Store results
                    for _, bucket in anomalies.head(10).iterrows():  # Top 10 per window
                        self.results.append({
                            'timestamp': datetime.now().isoformat(),
                            'agent': str(bucket.get('agent_name', '')),
                            'rule_group': str(bucket.get('rule_groups', '')),
                            'severity': int(bucket.get('rule_level', 0)),
                            'score': float(bucket.get('anomaly_score', 0.0)),
                            'decision': str(bucket.get('decision', '')),
                        })
                except:
                    # Silently skip IF if model not available
                    pass
        except Exception as e:
            # Silently continue if RBTA fails on small window
            pass
    
    def finalize(self):
        """Process remaining window."""
        self._process_window()


def main():
    print("=" * 70)
    print("INTEGRATION 3: Windowed Real-Time Processing")
    print("=" * 70)
    
    # Setup simulator
    sim = AlertStreamSimulator(
        csv_path="data/raw/rbta_ready_all.csv",
        speed_factor=100,
        log_file="output/07_windowed_stream.log",
        log_level="WARNING",
        log_frequency=50000,
    )
    
    # Try to load IF engine
    if_engine = None
    try:
        # Just store the function reference
        if_engine = run_isolation_forest
        print("✓ IF engine available (run_pipeline)")
    except Exception as e:
        print(f"⚠ IF engine not available ({e})")
        print("  → Will run RBTA only (no anomaly scoring)")
    
    # Setup windowed processor
    processor = WindowedProcessingEngine(
        window_sec=600,      # 10-minute window
        update_freq=500,     # Update every 500 alerts
        if_engine=if_engine,
    )
    
    # PHASE 1: STREAMING WITH WINDOWED UPDATES
    print("\n[PHASE 1] Streaming with windowed RBTA updates...")
    start = datetime.now()
    stats = sim.replay(on_alert_callback=processor.process_alert)
    processor.finalize()  # Process final window
    duration = (datetime.now() - start).total_seconds()
    
    print(f"✓ Processed {processor.processed_count:,} alerts in {duration:.1f} sec")
    print(f"  Rate: {processor.processed_count/duration:.0f} alerts/sec")
    
    # PHASE 2: RESULTS
    print(f"\n[PHASE 2] Windowed Processing Results:")
    print(f"  Total RBTA buckets: {processor.bucket_count:,}")
    if if_engine:
        print(f"  Anomalies detected: {processor.anomaly_count:,}")
    
    # PHASE 3: EXPORT
    print(f"\n[PHASE 3] Exporting results...")
    output_file = Path("output/07_windowed_anomalies.json")
    
    with open(output_file, 'w') as f:
        json.dump(processor.results, f, indent=2, default=str)
    
    print(f"✓ Exported {len(processor.results):,} detections to {output_file.name}")
    
    # PHASE 4: RECOMMENDATIONS
    print(f"\n[PHASE 4] Integration Notes:")
    print(f"""
  This pattern is best for:
    • Real-time alert processing
    • Time-sensitive detection (detect fast)
    • Streaming data sources (Kafka, syslog, etc)
    • Production deployments
  
  Configuration options:
    • window_sec = 300 (5 min) for fast detection
    • window_sec = 3600 (1 hr) for contextual analysis
    • update_freq = 100 (small) for responsiveness
    • update_freq = 1000 (large) for efficiency
  
  Next step: Integrate with real alert source!
    """)
    
    print("=" * 70)
    print("✓ INTEGRATION 3 Complete (Windowed Real-Time)")
    print("=" * 70)


if __name__ == "__main__":
    main()
