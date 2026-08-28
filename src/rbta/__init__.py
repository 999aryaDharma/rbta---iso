"""Rule-Based Temporal Aggregation (RBTA) Core Package.

Authoritative implementation of:
- Agent-local temporal state and EMA-based Elastic Time Window (ETW)
- Lossless bounded reorder buffer
- Single-bucket deterministic RBTA engine
"""

from src.rbta.engine import RBTAEngine, RBTAInvariantError
from src.rbta.reorder_buffer import LosslessReorderBuffer
from src.rbta.temporal_state import AgentTemporalState, TemporalStateError

__all__ = [
    "AgentTemporalState",
    "LosslessReorderBuffer",
    "RBTAEngine",
    "RBTAInvariantError",
    "TemporalStateError",
]
