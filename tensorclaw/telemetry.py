from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class MetricSample:
    timestamp_utc: str
    iteration: int
    phase: str
    metrics: dict[str, float]
    raw_line: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc or datetime.now(timezone.utc).isoformat(),
            "iteration": self.iteration,
            "phase": self.phase,
            "metrics": self.metrics,
            "raw_line": self.raw_line,
        }


def ensure_metrics_log(path: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.touch()


def append_metric_sample(path: str, sample: MetricSample) -> None:
    ensure_metrics_log(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample.as_dict(), ensure_ascii=True) + "\n")


def read_metric_samples(path: str, limit: int | None = None) -> list[MetricSample]:
    log_path = Path(path)
    if not log_path.exists():
        return []

    entries: list[MetricSample] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            metrics_payload = payload.get("metrics")
            if not isinstance(metrics_payload, dict):
                continue

            metrics: dict[str, float] = {}
            for key, value in metrics_payload.items():
                if not isinstance(key, str):
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if not metrics:
                continue

            try:
                iteration = int(payload.get("iteration", 0))
            except (TypeError, ValueError):
                iteration = 0

            phase = str(payload.get("phase", "experiment"))
            timestamp_utc = str(payload.get("timestamp_utc", ""))
            raw_line = str(payload.get("raw_line", ""))
            entries.append(
                MetricSample(
                    timestamp_utc=timestamp_utc,
                    iteration=iteration,
                    phase=phase,
                    metrics=metrics,
                    raw_line=raw_line,
                )
            )

    if limit is not None and limit > 0:
        return entries[-limit:]
    return entries


def build_series(samples: Iterable[MetricSample], max_points: int = 120) -> tuple[dict[str, list[float]], dict[str, float]]:
    series: dict[str, list[float]] = {}
    latest: dict[str, float] = {}
    for sample in samples:
        for name, value in sample.metrics.items():
            points = series.setdefault(name, [])
            points.append(value)
            if len(points) > max_points:
                del points[:-max_points]
            latest[name] = value
    return series, latest
