"""
src.streaming — Real-time alert streaming simulation
=====================================================
Module untuk simulate online alert streaming dari static CSV untuk testing
pipeline RBTA + IF dalam kondisi offline (development/research).

Main Components:
  - AlertStreamSimulator: Core streaming emulator
  - StreamStats: Statistics collector
  - setup_stream_logger: Logging configuration
"""

from .alert_stream_simulator import (
    AlertStreamSimulator,
    StreamStats,
    setup_stream_logger,
)

__all__ = [
    "AlertStreamSimulator",
    "StreamStats",
    "setup_stream_logger",
]
