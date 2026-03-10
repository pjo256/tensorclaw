from __future__ import annotations

import re
from typing import Optional


_FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        match = _FLOAT_PATTERN.search(value)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def extract_first_float(text: str, pattern: str) -> Optional[float]:
    compiled = re.compile(pattern, flags=re.MULTILINE)
    match = compiled.search(text)
    if not match:
        return None

    if match.lastindex:
        for index in range(1, match.lastindex + 1):
            value = _to_float(match.group(index))
            if value is not None:
                return value

    return _to_float(match.group(0))


def is_better(candidate: float, best: Optional[float], direction: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if direction == "minimize":
        return candidate <= (best - min_delta)
    return candidate >= (best + min_delta)


def evaluate_constraint(value: Optional[float], op: str, threshold: float) -> bool:
    if value is None:
        return False
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"Unsupported operator: {op}")
