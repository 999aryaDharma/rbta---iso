"""
04_realtime_callback.py  —  Real-Time Processing Callback
========================================================
Process alerts saat-saat mereka di-replay (online/streaming mode).

Gunakan untuk:
  - Real-time aggregation (count by agent, rule, severity)
  - Integration dengan detection engines (RBTA, IF)
  - Simulation pipeline testing
"""

import sys
import os
import collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.streaming import AlertStreamSimulator


class RealtimeAggregator:
    """Real-time aggregation of alerts during replay."""
    
    def __init__(self):
        self.agent_counts = collections.defaultdict(int)
        self.rulegroup_counts = collections.defaultdict(int)
        self.severity_counts = collections.defaultdict(int)
        self.total = 0
        self.high_severity_alerts = []
    
    def process_alert(self, alert):
        """Callback untuk setiap alert."""
        self.total += 1
        
        # Aggregate
        agent = alert.get('agent_name', 'unknown')
        rule_group = alert.get('rule_group_primary', 'unknown')
        severity = int(alert.get('rule_level', 1))
        
        self.agent_counts[agent] += 1
        self.rulegroup_counts[rule_group] += 1
        
        sev_bucket = 'L1-2' if severity <= 2 else 'L3-4' if severity <= 4 else 'L5-6' if severity <= 6 else 'L7+'
        self.severity_counts[sev_bucket] += 1
        
        # Track high-severity
        if severity >= 7:
            self.high_severity_alerts.append({
                'agent': agent,
                'rule': rule_group,
                'severity': severity,
                'timestamp': alert.get('timestamp_utc'),
            })
    
    def print_summary(self):
        """Print aggregation summary."""
        print("\n" + "-"*80)
        print("REAL-TIME AGGREGATION SUMMARY")
        print("-"*80)
        
        print(f"\nTotal alerts processed: {self.total:,}")
        
        print("\nTop agents:")
        for agent, count in sorted(self.agent_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = count / self.total * 100
            print(f"  {agent:<20}: {count:>6,} ({pct:>5.1f}%)")
        
        print("\nTop rule groups:")
        for rg, count in sorted(self.rulegroup_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = count / self.total * 100
            print(f"  {rg:<20}: {count:>6,} ({pct:>5.1f}%)")
        
        print("\nSeverity distribution:")
        for bucket in ['L1-2', 'L3-4', 'L5-6', 'L7+']:
            count = self.severity_counts.get(bucket, 0)
            pct = count / self.total * 100
            print(f"  {bucket:<20}: {count:>6,} ({pct:>5.1f}%)")
        
        print(f"\nHigh-severity alerts (L7+): {len(self.high_severity_alerts):,}")
        if self.high_severity_alerts:
            print("  Sample high-severity incidents:")
            for i, alert in enumerate(self.high_severity_alerts[:5], 1):
                print(f"    {i}. {alert['agent']} - {alert['rule']} (sev={alert['severity']})")
        
        print("-"*80 + "\n")


def main():
    print("\n" + "="*80)
    print("EXAMPLE 4: Real-Time Callback Processing")
    print("="*80 + "\n")
    
    # Initialize aggregator
    agg = RealtimeAggregator()
    
    # Initialize simulator
    sim = AlertStreamSimulator(
        csv_path="data/raw/rbta_ready_all.csv",
        speed_factor=100,
        log_file="output/04_realtime.log",
        log_level="INFO",
        log_frequency=10000,
    )
    
    # Register callback
    sim.on_alert(agg.process_alert)
    
    print("Running real-time aggregation...\n")
    stats = sim.replay()
    
    # Print aggregation results
    agg.print_summary()
    
    print("="*80)
    print("REAL-TIME PROCESSING COMPLETE")
    print("="*80)
    print(f"Total alerts: {stats.total_alerts:,}")
    print(f"Duration: {stats.duration_sec:.1f}s")
    print(f"Processing rate: {stats.alerts_per_sec:.0f} alerts/sec")
    print(f"Log file: output/04_realtime.log")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
