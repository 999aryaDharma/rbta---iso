"""
03_filtering.py  —  Replay dengan Filtering (Agents, Severity)
==============================================================
Simulasi hanya specific agents atau severity levels untuk focused testing.

Gunakan untuk:
  - Test specific scenarios (e.g., only high-severity dari specific agent)
  - Reduce dataset size untuk faster iteration
  - Focus development effort
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.streaming import AlertStreamSimulator


def test_scenario(name, **kwargs):
    """Helper untuk run scenario dengan specific config."""
    print(f"\n{'='*80}")
    print(f"Scenario: {name}")
    print(f"{'='*80}")
    
    sim = AlertStreamSimulator(
        csv_path="data/raw/rbta_ready_all.csv",
        log_level="WARNING",  # Less verbose
        log_frequency=50000,   # Log less frequently
        **kwargs
    )
    
    stats = sim.replay()
    
    print(f"\nResults:")
    print(f"  Alerts: {stats.total_alerts:,}")
    print(f"  Duration: {stats.duration_sec:.1f}s")
    print(f"  Rate: {stats.alerts_per_sec:.0f}/sec")
    
    return stats


def main():
    print("\n" + "="*80)
    print("EXAMPLE 3: Alert Stream Filtering")
    print("="*80)
    
    # Scenario 1: Only high-severity alerts
    test_scenario(
        "High-severity only (level 7+)",
        speed_factor=100,
        filter_severity_min=7,
        log_file="output/03_scenario1.log",
    )
    
    # Scenario 2: Only specific agents
    test_scenario(
        "Only DFIR-IRIS and PusatKarir",
        speed_factor=100,
        filter_agents=["dfir-iris", "pusatkarir"],
        log_file="output/03_scenario2.log",
    )
    
    # Scenario 3: Combination: high-severity from specific agents
    test_scenario(
        "High-severity (7+) from dfir-iris",
        speed_factor=100,
        filter_agents=["dfir-iris"],
        filter_severity_min=7,
        log_file="output/03_scenario3.log",
    )
    
    print("\n" + "="*80)
    print("All scenarios completed!")
    print(f"Logs saved in output/03_scenario*.log")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
