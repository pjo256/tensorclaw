from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True, kw_only=True)
class EngineEvent:
    kind: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class StartupEvent(EngineEvent):
    stage: str
    message: str
    kind: str = field(default="startup", init=False)


@dataclass(slots=True)
class DiscoveryEvent(EngineEvent):
    message: str
    kind: str = field(default="discovery", init=False)


@dataclass(slots=True)
class ChatEvent(EngineEvent):
    role: Literal["user", "assistant", "system"]
    content: str
    kind: str = field(default="chat", init=False)


@dataclass(slots=True)
class ChatStreamEvent(EngineEvent):
    role: Literal["assistant", "system"]
    mode: Literal["start", "delta", "end"]
    text: str = ""
    kind: str = field(default="chat_stream", init=False)


@dataclass(slots=True)
class ProposalEvent(EngineEvent):
    title: str
    body: str
    kind: str = field(default="proposal", init=False)


@dataclass(slots=True)
class IterationEvent(EngineEvent):
    iteration: int
    status: str
    message: str
    commit: str = ""
    kind: str = field(default="iteration", init=False)


@dataclass(slots=True)
class MetricEvent(EngineEvent):
    metric_name: str
    direction: str
    value: float | None
    best: float | None
    baseline: float | None
    kind: str = field(default="metric", init=False)


@dataclass(slots=True)
class TelemetryEvent(EngineEvent):
    iteration: int
    phase: str
    metrics: dict[str, float]
    kind: str = field(default="telemetry", init=False)


@dataclass(slots=True)
class UsageEvent(EngineEvent):
    source: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    kind: str = field(default="usage", init=False)


@dataclass(slots=True)
class OutputLineEvent(EngineEvent):
    phase: str
    stream: Literal["stdout", "stderr", "meta"]
    line: str
    kind: str = field(default="output", init=False)


@dataclass(slots=True)
class DecisionEvent(EngineEvent):
    iteration: int
    status: str
    metric: float | None
    best: float | None
    commit: str
    kind: str = field(default="decision", init=False)


@dataclass(slots=True)
class ErrorEvent(EngineEvent):
    message: str
    detail: str = ""
    kind: str = field(default="error", init=False)
