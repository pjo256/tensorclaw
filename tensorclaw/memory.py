from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "is",
    "are",
    "be",
    "by",
    "as",
    "at",
    "it",
    "that",
    "this",
    "from",
    "run",
    "iteration",
    "metric",
}


@dataclass(slots=True)
class MemoryEntry:
    timestamp_utc: str
    iteration: int
    status: str
    metric_name: str
    metric_value: float | None
    commit: str
    summary: str
    tags: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc or datetime.now(timezone.utc).isoformat(),
            "iteration": self.iteration,
            "status": self.status,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "commit": self.commit,
            "summary": self.summary,
            "tags": self.tags,
        }


def ensure_memory_log(path: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.touch()


def append_memory_entry(path: str, entry: MemoryEntry) -> None:
    ensure_memory_log(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.as_dict(), ensure_ascii=True) + "\n")


def read_memory_entries(path: str, limit: int | None = None) -> list[MemoryEntry]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    entries: list[MemoryEntry] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
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
            try:
                iteration = int(payload.get("iteration", 0))
            except (TypeError, ValueError):
                iteration = 0
            metric_name = str(payload.get("metric_name", ""))
            metric_value_raw = payload.get("metric_value")
            metric_value: float | None
            try:
                metric_value = float(metric_value_raw) if metric_value_raw is not None else None
            except (TypeError, ValueError):
                metric_value = None

            tags_payload = payload.get("tags", [])
            tags = [str(item) for item in tags_payload] if isinstance(tags_payload, list) else []
            entries.append(
                MemoryEntry(
                    timestamp_utc=str(payload.get("timestamp_utc", "")),
                    iteration=iteration,
                    status=str(payload.get("status", "")),
                    metric_name=metric_name,
                    metric_value=metric_value,
                    commit=str(payload.get("commit", "")),
                    summary=str(payload.get("summary", "")).strip(),
                    tags=tags,
                )
            )

    if limit is not None and limit > 0:
        return entries[-limit:]
    return entries


def infer_tags(*texts: str, max_tags: int = 12) -> list[str]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in TOKEN_RE.findall(text.lower()):
            if len(token) < 3 or token in STOP_WORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_tags]]


def format_memory_entries(entries: Iterable[MemoryEntry]) -> str:
    lines: list[str] = []
    for entry in entries:
        metric_text = "n/a" if entry.metric_value is None else f"{entry.metric_value:.6f}"
        lines.append(
            f"iter {entry.iteration} | {entry.status} | {entry.metric_name}={metric_text} | {entry.summary}"
        )
    return "\n".join(lines) if lines else "No durable project memory yet."


def retrieve_relevant(entries: list[MemoryEntry], query: str, limit: int = 6) -> list[MemoryEntry]:
    query_tokens = {token for token in TOKEN_RE.findall(query.lower()) if token not in STOP_WORDS and len(token) >= 3}
    if not entries:
        return []
    if not query_tokens:
        return entries[-limit:]

    scored: list[tuple[float, MemoryEntry]] = []
    total = len(entries)
    for idx, entry in enumerate(entries):
        tags = set(entry.tags)
        overlap = len(tags.intersection(query_tokens))
        if overlap == 0:
            # keep slight recency bias for non-overlapping memory
            recency = (idx + 1) / max(1, total)
            score = 0.1 * recency
        else:
            recency = (idx + 1) / max(1, total)
            score = float(overlap) + 0.25 * recency
        scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [entry for score, entry in scored if score > 0][:limit]
    return selected if selected else entries[-limit:]
