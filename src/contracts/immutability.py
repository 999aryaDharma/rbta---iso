"""Defensive recursive freezing utilities for canonical data contracts."""
from types import MappingProxyType
from typing import Any, Mapping


def freeze_value(val: Any) -> Any:
    """Recursively convert nested mutable data structures into immutable types.

    Conversions:
    - Mapping / dict -> MappingProxyType where all values are recursively frozen.
    - list / tuple -> tuple where all elements are recursively frozen.
    - set / frozenset -> frozenset where all elements are recursively frozen.
    - Primitives and already-immutable objects are returned as-is.

    Parameters
    ----------
    val : Any
        Input value to freeze recursively.

    Returns
    -------
    Any
        Deeply immutable representation of the input.
    """
    if isinstance(val, Mapping):
        return MappingProxyType({k: freeze_value(v) for k, v in val.items()})
    elif isinstance(val, (list, tuple)):
        return tuple(freeze_value(item) for item in val)
    elif isinstance(val, (set, frozenset)):
        return frozenset(freeze_value(item) for item in val)
    return val
