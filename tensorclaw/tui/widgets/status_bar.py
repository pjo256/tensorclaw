from __future__ import annotations

from textual.widgets import Static


class StatusBar(Static):
    def __init__(self) -> None:
        super().__init__(id="status-bar")

    def update_status(
        self,
        *,
        phase: str,
        metric_name: str,
        direction: str,
        best: float | None,
        iteration: int,
        has_pending_plan: bool,
        busy: bool,
        monitor: str = "",
        tokens: str = "",
    ) -> None:
        best_text = "n/a" if best is None else f"{best:.6f}"
        pending = "plan pending" if has_pending_plan else "no pending plan"
        mode = "busy" if busy else "ready"
        hints = "Enter chat | Y approve | N cancel | Ctrl+C quit"
        monitor_text = f" | live: {monitor}" if monitor else ""
        token_text = f" | {tokens}" if tokens else ""
        self.update(
            f"state={phase} | mode={mode} | metric={metric_name} ({direction}) | best={best_text} | iter={iteration} | {pending}{monitor_text}{token_text} | {hints}"
        )
