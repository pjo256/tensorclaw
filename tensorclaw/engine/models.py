from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..spec import ResearchSpec


@dataclass(slots=True)
class ChatMessage:
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class ProposalPlan:
    hypothesis: str
    planned_edits: list[str]
    expected_impact: str
    risk: str
    direction: str = ""
    raw_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        edits = "\n".join(f"- {item}" for item in self.planned_edits) if self.planned_edits else "- (none)"
        return (
            f"Hypothesis: {self.hypothesis}\n\n"
            f"Planned edits:\n{edits}\n\n"
            f"Expected impact: {self.expected_impact}\n"
            f"Risk: {self.risk}"
        )


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    status: str
    metric_value: float | None
    commit: str
    description: str = ""
    is_best: bool = False
    is_latest: bool = False


@dataclass(slots=True)
class MetricSeriesPoint:
    iteration: int
    value: float
    status: str


@dataclass(slots=True)
class MemoryRecord:
    timestamp_utc: str
    iteration: int
    status: str
    metric_name: str
    metric_value: float | None
    commit: str
    summary: str
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PendingAction:
    type: Literal["proposal"]
    plan: ProposalPlan


@dataclass(slots=True)
class AppState:
    project_root: Path
    spec: ResearchSpec | None = None
    phase: str = "startup"
    metric_name: str = ""
    metric_direction: str = "minimize"
    metric_baseline: float | None = None
    best_metric: float | None = None
    anchor_commit: str = "workspace"
    current_iteration: int = 0
    running_iteration: int | None = None
    running_commit: str = ""
    running_phase: str = "idle"
    current_metric: float | None = None
    chat_history: list[ChatMessage] = field(default_factory=list)
    iteration_records: list[IterationRecord] = field(default_factory=list)
    metric_series: list[MetricSeriesPoint] = field(default_factory=list)
    telemetry_series: dict[str, list[float]] = field(default_factory=dict)
    telemetry_latest: dict[str, float] = field(default_factory=dict)
    live_monitor: dict[str, float] = field(default_factory=dict)
    memory_records: list[MemoryRecord] = field(default_factory=list)
    token_input_total: int = 0
    token_output_total: int = 0
    token_total: int = 0
    last_input_tokens: int | None = None
    last_output_tokens: int | None = None
    last_total_tokens: int | None = None
    last_token_source: str = ""
    pending_action: PendingAction | None = None
    output_lines: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def has_pending_plan(self) -> bool:
        return self.pending_action is not None and self.pending_action.type == "proposal"
