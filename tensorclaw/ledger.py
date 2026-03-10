from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


HEADER = [
    "timestamp_utc",
    "iteration",
    "commit",
    "metric_name",
    "metric_value",
    "memory_gb",
    "status",
    "description",
    "constraints",
    "log_path",
]


@dataclass
class LedgerRow:
    iteration: int
    commit: str
    metric_name: str
    metric_value: float
    memory_gb: float
    status: str
    description: str
    constraints: str = ""
    log_path: str = ""
    timestamp_utc: str = ""

    def as_dict(self) -> dict[str, str]:
        timestamp = self.timestamp_utc or datetime.now(timezone.utc).isoformat()
        return {
            "timestamp_utc": timestamp,
            "iteration": str(self.iteration),
            "commit": self.commit,
            "metric_name": self.metric_name,
            "metric_value": f"{self.metric_value:.6f}",
            "memory_gb": f"{self.memory_gb:.1f}",
            "status": self.status,
            "description": _sanitize(self.description),
            "constraints": _sanitize(self.constraints),
            "log_path": self.log_path,
        }


def _sanitize(value: str) -> str:
    return (value or "").replace("\t", " ").replace("\n", " ").strip()


def ensure_ledger(path: str) -> None:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        return
    with ledger_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()


def append_row(path: str, row: LedgerRow) -> None:
    ensure_ledger(path)
    with Path(path).open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writerow(row.as_dict())


def read_rows(path: str) -> List[dict[str, str]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    with ledger_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def best_metric(rows: Iterable[dict[str, str]], direction: str) -> Optional[float]:
    values = []
    for row in rows:
        if row.get("status") != "keep":
            continue
        try:
            values.append(float(row.get("metric_value", "")))
        except ValueError:
            continue
    if not values:
        return None
    return min(values) if direction == "minimize" else max(values)


def summarize_recent(rows: list[dict[str, str]], limit: int = 5) -> str:
    if not rows:
        return "No prior experiments."
    selected = rows[-limit:]
    lines = []
    for row in selected:
        lines.append(
            f"iter {row.get('iteration')} | {row.get('status')} | "
            f"{row.get('metric_name')}={row.get('metric_value')} | {row.get('description')}"
        )
    return "\n".join(lines)
