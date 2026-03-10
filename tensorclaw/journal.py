from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def ensure_journal(path: str, title: str) -> None:
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    if journal_path.exists():
        return

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with journal_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title} Journal\n\n")
        f.write(f"Created: {created}\n\n")


def append_entry(
    path: str,
    iteration: int,
    idea: str,
    status: str,
    metric_name: str,
    metric_value: Optional[float],
    commit: str,
    log_path: str,
    note: str = "",
    instruction_path: str = "",
) -> None:
    ensure_journal(path, "TensorClaw")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    metric = "n/a" if metric_value is None else f"{metric_value:.6f}"

    with Path(path).open("a", encoding="utf-8") as f:
        f.write(f"## Iteration {iteration} ({ts})\n\n")
        f.write(f"- Status: `{status}`\n")
        f.write(f"- Commit: `{commit}`\n")
        f.write(f"- Idea: {idea}\n")
        f.write(f"- Metric: `{metric_name}={metric}`\n")
        if instruction_path:
            f.write(f"- Instruction: `{instruction_path}`\n")
        if log_path:
            f.write(f"- Log: `{log_path}`\n")
        if note:
            f.write(f"- Notes: {note}\n")
        f.write("\n")
