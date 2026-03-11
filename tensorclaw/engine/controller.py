from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from ..git_ops import get_state
from ..journal import append_entry, ensure_journal
from ..ledger import LedgerRow, append_row, best_metric, ensure_ledger, read_rows, summarize_recent
from ..memory import MemoryEntry, append_memory_entry, ensure_memory_log, format_memory_entries, infer_tags, read_memory_entries, retrieve_relevant
from ..shell import run_command
from ..spec import ResearchSpec
from ..telemetry import MetricSample, append_metric_sample, build_series, ensure_metrics_log, read_metric_samples
from ..templates import render_template, shell_escape
from .bootstrap import load_spec_for_project
from .events import (
    ChatEvent,
    ChatStreamEvent,
    DecisionEvent,
    EngineEvent,
    ErrorEvent,
    IterationEvent,
    MetricEvent,
    ProposalEvent,
    StartupEvent,
    TelemetryEvent,
    UsageEvent,
)
from .models import AppState, ChatMessage, IterationRecord, MemoryRecord, MetricSeriesPoint, PendingAction, ProposalPlan
from .runner import IterationRunner


CHAT_HISTORY_LIMIT = 8
MEMORY_CONTEXT_LIMIT = 6
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
AGENT_MODE_FLAG_RE = re.compile(r"(--mode\s+)([^\s]+)")
CHAT_INTENTS = {"chat_only", "proposal_ready"}
PENDING_ACTIONS = {"approve", "reject", "chat_only"}
USAGE_CONTAINER_KEYS = {"usage", "tokenusage", "usage_metadata", "usagemetadata", "token_usage"}

CHAT_DECISION_PROMPT = """You are TensorClaw's research copilot.

Return ONLY a JSON object with:
{
  "assistant_reply": "short answer for the user",
  "intent": "chat_only | proposal_ready",
  "proposal": {
    "hypothesis": "one sentence",
    "planned_edits": ["file: change", "..."],
    "expected_impact": "one sentence",
    "risk": "one sentence",
    "direction": "optional short direction"
  }
}

Rules:
- If the user asks a question, explains context, or seeks clarification: intent MUST be "chat_only".
- Only set intent="proposal_ready" when the user explicitly asks to run/try an experiment now.
- Do not start execution. TensorClaw will wait for explicit user approval.
- Keep assistant_reply concise and actionable.
- Do NOT start assistant_reply with canned filler (e.g., "Great", "Sure", "Absolutely", "Sounds good", "Understood").
- Return valid JSON only.
"""

PENDING_DECISION_PROMPT = """You are TensorClaw's pending-plan arbiter.

There is already a pending experiment proposal. Read the latest user message and decide if they want to:
- approve the pending proposal now
- reject/cancel the pending proposal
- continue chatting without changing proposal state

Return ONLY a JSON object:
{
  "assistant_reply": "short natural response to the user",
  "action": "approve | reject | chat_only"
}

Rules:
- Infer intent semantically from user language, not by rigid keyword matching.
- If user asks a question or asks for changes/explanation, action MUST be "chat_only".
- Use "approve" only when user clearly indicates to proceed now.
- Use "reject" only when user clearly indicates to cancel/stop this plan.
- Do NOT start assistant_reply with canned filler (e.g., "Great", "Sure", "Absolutely", "Sounds good", "Understood").
- Return valid JSON only.
"""


EventSink = Callable[[EngineEvent], None] | None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_metric_key(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("/", "_per_")
    normalized = normalized.replace("%", "_pct")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("empty output")

    candidates: list[str] = [text]
    candidates.extend(match.group(1).strip() for match in JSON_BLOCK_RE.finditer(text))

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first : last + 1].strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError("no valid JSON object")


def _sanitize_json_line(raw_line: str) -> str:
    text = CONTROL_CHARS_RE.sub("", raw_line).strip()
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return ""
    return text[start : end + 1]


def _message_text(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def _extract_payload_from_json_stream(raw_text: str) -> dict[str, Any] | None:
    lines = raw_text.splitlines()
    if not lines:
        return None

    assistant_text_candidates: list[str] = []
    for line in lines:
        line = _sanitize_json_line(line)
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type == "turn_end":
            text = _message_text(event.get("message"))
            if text:
                assistant_text_candidates.append(text)
        elif event_type == "agent_end":
            messages = event.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        text = _message_text(message)
                        if text:
                            assistant_text_candidates.append(text)
                            break

    for text in reversed(assistant_text_candidates):
        try:
            return _extract_json_object(text)
        except ValueError:
            continue
    return None


def _collect_delta_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_collect_delta_text(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("delta", "text", "content", "output_text"):
            candidate = value.get(key)
            text = _collect_delta_text(candidate)
            if text:
                return text
        return ""
    return ""


def _extract_stream_delta_from_json_line(raw_line: str) -> str:
    line = _sanitize_json_line(raw_line)
    if not line:
        return ""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if not isinstance(event, dict):
        return ""

    event_type = str(event.get("type", "")).lower()
    if event_type == "message_update":
        message_event = event.get("assistantMessageEvent")
        if isinstance(message_event, dict):
            message_event_type = str(message_event.get("type", "")).lower()
            if "delta" in message_event_type:
                text = _collect_delta_text(message_event.get("delta"))
                if text:
                    return text
            # Fallback for providers that place text on alternate keys.
            text = _collect_delta_text(message_event.get("text"))
            if text:
                return text
    if "delta" not in event_type and "chunk" not in event_type and "partial" not in event_type:
        return ""

    for key in ("delta", "text", "content", "output_text", "event", "data", "response"):
        text = _collect_delta_text(event.get(key))
        if text:
            return text
    return ""


def _extract_assistant_reply_candidate(stream_text: str) -> str | None:
    marker = '"assistant_reply"'
    marker_index = stream_text.find(marker)
    if marker_index == -1:
        return None
    colon_index = stream_text.find(":", marker_index + len(marker))
    if colon_index == -1:
        return None

    pos = colon_index + 1
    while pos < len(stream_text) and stream_text[pos].isspace():
        pos += 1
    if pos >= len(stream_text) or stream_text[pos] != '"':
        return None
    pos += 1

    chars: list[str] = []
    escape = False
    while pos < len(stream_text):
        ch = stream_text[pos]
        if escape:
            chars.append(ch)
            escape = False
            pos += 1
            continue
        if ch == "\\":
            escape = True
            pos += 1
            continue
        if ch == '"':
            return "".join(chars)
        chars.append(ch)
        pos += 1

    # Return partial value while string is still open.
    return "".join(chars) if chars else None


def _usage_bucket(key: str) -> str | None:
    lower = key.strip().lower()
    if "token" not in lower:
        return None
    if lower.startswith("max_"):
        return None
    if "input" in lower or "prompt" in lower:
        return "input"
    if "output" in lower or "completion" in lower:
        return "output"
    if "total" in lower:
        return "total"
    return None


def _collect_usage_from_obj(value: Any, out: dict[str, int]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_usage_from_obj(item, out)
        return
    if not isinstance(value, dict):
        return

    for key, nested in value.items():
        if isinstance(nested, (int, float)):
            bucket = _usage_bucket(str(key))
            if bucket is None:
                continue
            parsed = int(nested)
            if parsed < 0:
                continue
            prev = out.get(bucket)
            if prev is None or parsed > prev:
                out[bucket] = parsed
        elif isinstance(nested, (dict, list)):
            _collect_usage_from_obj(nested, out)


def _extract_usage_from_event(event: dict[str, Any]) -> dict[str, int]:
    usage: dict[str, int] = {}

    containers: list[Any] = []
    for key, value in event.items():
        normalized = re.sub(r"[^a-z0-9_]+", "", str(key).strip().lower())
        if normalized in USAGE_CONTAINER_KEYS:
            containers.append(value)

    if containers:
        for container in containers:
            _collect_usage_from_obj(container, usage)
    else:
        _collect_usage_from_obj(event, usage)

    input_tokens = usage.get("input")
    output_tokens = usage.get("output")
    if usage.get("total") is None and (input_tokens is not None or output_tokens is not None):
        usage["total"] = (input_tokens or 0) + (output_tokens or 0)
    return usage


def _extract_usage_from_json_line(raw_line: str) -> dict[str, int]:
    line = _sanitize_json_line(raw_line)
    if not line:
        return {}
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return {}
    if not isinstance(event, dict):
        return {}
    return _extract_usage_from_event(event)


def _merge_usage(dest: dict[str, int], src: dict[str, int]) -> None:
    for key in ("input", "output", "total"):
        value = src.get(key)
        if value is None:
            continue
        prev = dest.get(key)
        if prev is None or value > prev:
            dest[key] = value


def _ensure_json_mode(agent_command: str) -> str:
    if AGENT_MODE_FLAG_RE.search(agent_command):
        return AGENT_MODE_FLAG_RE.sub(lambda m: f"{m.group(1)}json", agent_command, count=1)
    return f"{agent_command} --mode json"


def _has_agent_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _parse_chat_payload(payload: dict[str, Any]) -> tuple[str, str, ProposalPlan | None]:
    assistant_reply = str(payload.get("assistant_reply", "")).strip() or "I can help with the next experiment step."
    intent = str(payload.get("intent", "chat_only")).strip().lower()
    if intent not in CHAT_INTENTS:
        intent = "chat_only"

    proposal_raw = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else {}
    proposal: ProposalPlan | None = None
    if intent == "proposal_ready":
        hypothesis = str(proposal_raw.get("hypothesis", "")).strip() or "Try one focused experiment that can improve the objective metric."
        edits = proposal_raw.get("planned_edits", [])
        planned_edits = [str(item).strip() for item in edits] if isinstance(edits, list) else []
        expected_impact = str(proposal_raw.get("expected_impact", "")).strip() or "Potentially improve the objective metric."
        risk = str(proposal_raw.get("risk", "")).strip() or "May regress the objective metric."
        direction = str(proposal_raw.get("direction", "")).strip()
        proposal = ProposalPlan(
            hypothesis=hypothesis,
            planned_edits=planned_edits,
            expected_impact=expected_impact,
            risk=risk,
            direction=direction,
            raw_text=json.dumps(proposal_raw, ensure_ascii=True),
        )
    return assistant_reply, intent, proposal


def _parse_pending_payload(payload: dict[str, Any]) -> tuple[str, str]:
    assistant_reply = str(payload.get("assistant_reply", "")).strip() or "Understood."
    action = str(payload.get("action", "chat_only")).strip().lower()
    if action not in PENDING_ACTIONS:
        action = "chat_only"
    return assistant_reply, action


def _parse_proposal_from_text(text: str) -> ProposalPlan:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    hypothesis = "Try one focused experiment that can improve the objective metric."
    expected_impact = "Potentially improve the objective metric."
    risk = "May regress the objective metric."
    edits: list[str] = []

    for line in lines:
        lower = line.lower()
        if lower.startswith("plan:"):
            hypothesis = line.split(":", 1)[1].strip() or hypothesis
        elif lower.startswith("impact:"):
            expected_impact = line.split(":", 1)[1].strip() or expected_impact
        elif lower.startswith("risk:"):
            risk = line.split(":", 1)[1].strip() or risk
        elif line.startswith("-"):
            edits.append(line.lstrip("- ").strip())

    return ProposalPlan(
        hypothesis=hypothesis,
        planned_edits=edits,
        expected_impact=expected_impact,
        risk=risk,
        raw_text=text,
    )


class ResearchController:
    def __init__(self, project_root: Path, dry_run: bool = False, event_sink: EventSink = None) -> None:
        self.state = AppState(project_root=project_root, dry_run=dry_run)
        self._event_sink = event_sink
        self._runner = IterationRunner()
        self._lock = threading.RLock()

    def _emit(self, event: EngineEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    def _runner_emit(self, event: EngineEvent) -> None:
        with self._lock:
            if isinstance(event, TelemetryEvent):
                metric_key = _normalize_metric_key(self.state.metric_name) if self.state.metric_name else ""
                for key, value in event.metrics.items():
                    if not isinstance(key, str):
                        continue
                    values = self.state.telemetry_series.setdefault(key, [])
                    values.append(float(value))
                    if len(values) > 120:
                        del values[:-120]
                    self.state.telemetry_latest[key] = float(value)
                    self.state.live_monitor[key] = float(value)
                    if key == self.state.metric_name or (metric_key and key == metric_key):
                        self.state.current_metric = float(value)
                    if len(self.state.live_monitor) > 24:
                        # Keep the monitor map compact and predictable.
                        keys = list(self.state.live_monitor.keys())
                        for stale in keys[:-24]:
                            self.state.live_monitor.pop(stale, None)
            elif isinstance(event, MetricEvent):
                if event.value is not None:
                    self.state.live_monitor[event.metric_name] = float(event.value)
                    self.state.current_metric = float(event.value)
            elif isinstance(event, IterationEvent):
                self.state.running_iteration = event.iteration
                self.state.running_phase = event.status or "running"
                if event.iteration > self.state.current_iteration:
                    self.state.current_iteration = event.iteration
                if event.commit:
                    self.state.running_commit = event.commit
            elif isinstance(event, DecisionEvent):
                self.state.running_phase = "idle"
                self.state.running_iteration = None
                self.state.running_commit = ""
        self._emit(event)

    def _record_usage(self, source: str, usage: dict[str, int]) -> None:
        input_tokens = usage.get("input")
        output_tokens = usage.get("output")
        total_tokens = usage.get("total")
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return

        with self._lock:
            self.state.last_token_source = source
            self.state.last_input_tokens = input_tokens
            self.state.last_output_tokens = output_tokens
            if total_tokens is None and (input_tokens is not None or output_tokens is not None):
                total_tokens = (input_tokens or 0) + (output_tokens or 0)
            self.state.last_total_tokens = total_tokens

            if input_tokens is not None:
                self.state.token_input_total += max(0, int(input_tokens))
            if output_tokens is not None:
                self.state.token_output_total += max(0, int(output_tokens))
            if total_tokens is not None:
                self.state.token_total += max(0, int(total_tokens))
            else:
                self.state.token_total += max(0, int(input_tokens or 0)) + max(0, int(output_tokens or 0))

        self._emit(
            UsageEvent(
                source=source,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        )

    @property
    def session_path(self) -> Path:
        return self.state.project_root / ".tensorclaw" / "session.jsonl"

    def _tensorclaw_dir(self) -> Path:
        if self.state.spec is not None:
            return Path(self.state.spec.paths.results_tsv).resolve().parent
        return self.state.project_root / ".tensorclaw"

    @property
    def metrics_log_path(self) -> Path:
        return self._tensorclaw_dir() / "metrics.jsonl"

    @property
    def memory_log_path(self) -> Path:
        return self._tensorclaw_dir() / "memory.jsonl"

    def _append_session_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            path = self.session_path
            path.parent.mkdir(parents=True, exist_ok=True)
            event = {"kind": kind, "payload": payload}
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=True) + "\n")

    def _load_session_events(self) -> None:
        path = self.session_path
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return

        pending: ProposalPlan | None = None
        chat_history: list[ChatMessage] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("kind")
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            if kind == "chat":
                role = str(payload.get("role", "assistant"))
                content = str(payload.get("content", ""))
                if role in {"user", "assistant", "system"} and content:
                    chat_history.append(ChatMessage(role=role, content=content))
            elif kind == "proposal":
                hypothesis = str(payload.get("hypothesis", "")).strip()
                if not hypothesis:
                    continue
                edits = payload.get("planned_edits", [])
                if not isinstance(edits, list):
                    edits = []
                pending = ProposalPlan(
                    hypothesis=hypothesis,
                    planned_edits=[str(item) for item in edits],
                    expected_impact=str(payload.get("expected_impact", "Potentially improve the objective metric.")),
                    risk=str(payload.get("risk", "May regress the objective metric.")),
                    direction=str(payload.get("direction", "")),
                )
            elif kind == "proposal_cleared":
                pending = None

        with self._lock:
            self.state.chat_history = chat_history[-200:]
            if pending is not None:
                self.state.pending_action = PendingAction(type="proposal", plan=pending)

    def _refresh_iteration_state(self) -> None:
        spec = self.state.spec
        if spec is None:
            return
        rows = read_rows(spec.paths.results_tsv)
        records: list[IterationRecord] = []
        points: list[MetricSeriesPoint] = []
        best = best_metric(rows, spec.metric.direction)

        best_keep_iteration = None
        best_value = None
        for row in rows:
            if row.get("status") != "keep":
                continue
            value = _safe_float(row.get("metric_value"))
            if value is None:
                continue
            if best_value is None:
                best_value = value
                best_keep_iteration = row.get("iteration")
            elif spec.metric.direction == "maximize" and value > best_value:
                best_value = value
                best_keep_iteration = row.get("iteration")
            elif spec.metric.direction != "maximize" and value < best_value:
                best_value = value
                best_keep_iteration = row.get("iteration")

        for idx, row in enumerate(rows):
            iter_text = row.get("iteration", "0")
            try:
                iteration = int(iter_text)
            except ValueError:
                continue
            status = row.get("status", "unknown")
            metric_value = _safe_float(row.get("metric_value"))
            commit = row.get("commit", "")
            description = row.get("description", "")
            is_latest = idx == len(rows) - 1
            is_best = status == "keep" and iter_text == best_keep_iteration
            records.append(
                IterationRecord(
                    iteration=iteration,
                    status=status,
                    metric_value=metric_value,
                    commit=commit,
                    description=description,
                    is_best=is_best,
                    is_latest=is_latest,
                )
            )
            if metric_value is not None and status in {"keep", "discard"}:
                points.append(MetricSeriesPoint(iteration=iteration, value=metric_value, status=status))

        with self._lock:
            self.state.iteration_records = records
            self.state.metric_series = points
            self.state.best_metric = best
            self.state.current_metric = points[-1].value if points else None
            if records:
                self.state.current_iteration = max(item.iteration for item in records)

    def _refresh_telemetry_state(self) -> None:
        samples = read_metric_samples(str(self.metrics_log_path))
        series, latest = build_series(samples, max_points=120)
        with self._lock:
            self.state.telemetry_series = series
            self.state.telemetry_latest = latest
            self.state.live_monitor = dict(latest)

    def _refresh_memory_state(self) -> None:
        entries = read_memory_entries(str(self.memory_log_path), limit=400)
        records: list[MemoryRecord] = []
        for entry in entries:
            records.append(
                MemoryRecord(
                    timestamp_utc=entry.timestamp_utc,
                    iteration=entry.iteration,
                    status=entry.status,
                    metric_name=entry.metric_name,
                    metric_value=entry.metric_value,
                    commit=entry.commit,
                    summary=entry.summary,
                    tags=entry.tags,
                )
            )
        with self._lock:
            self.state.memory_records = records

    def _backfill_metrics_from_ledger(self) -> None:
        if self.metrics_log_path.exists() and self.metrics_log_path.stat().st_size > 0:
            return
        spec = self.state.spec
        if spec is None:
            return
        rows = read_rows(spec.paths.results_tsv)
        metric_key = _normalize_metric_key(spec.metric.name)
        for row in rows:
            metric_value = _safe_float(row.get("metric_value"))
            if metric_value is None:
                continue
            try:
                iteration = int(row.get("iteration", "0"))
            except ValueError:
                iteration = 0
            metrics: dict[str, float] = {metric_key: metric_value}
            memory_gb = _safe_float(row.get("memory_gb"))
            if memory_gb is not None and memory_gb > 0:
                metrics["memory_gb"] = memory_gb
            append_metric_sample(
                str(self.metrics_log_path),
                MetricSample(
                    timestamp_utc=row.get("timestamp_utc", "") or "",
                    iteration=iteration,
                    phase="backfill",
                    metrics=metrics,
                    raw_line=f"status={row.get('status', '')}",
                ),
            )

    def _backfill_memory_from_ledger(self) -> None:
        if self.memory_log_path.exists() and self.memory_log_path.stat().st_size > 0:
            return
        spec = self.state.spec
        if spec is None:
            return
        rows = read_rows(spec.paths.results_tsv)
        for row in rows:
            try:
                iteration = int(row.get("iteration", "0"))
            except ValueError:
                iteration = 0
            metric_value = _safe_float(row.get("metric_value"))
            status = str(row.get("status", "")).strip() or "unknown"
            commit = str(row.get("commit", "")).strip()
            description = str(row.get("description", "")).strip()
            if iteration == 0 and description.lower() == "baseline":
                status = "baseline"
            summary = (
                f"idea={description or '(none)'}; status={status}; {spec.metric.name}="
                f"{'n/a' if metric_value is None else f'{metric_value:.6f}'}"
            )
            append_memory_entry(
                str(self.memory_log_path),
                MemoryEntry(
                    timestamp_utc=row.get("timestamp_utc", "") or "",
                    iteration=iteration,
                    status=status,
                    metric_name=spec.metric.name,
                    metric_value=metric_value,
                    commit=commit,
                    summary=summary,
                    tags=infer_tags(description, status, spec.metric.name),
                ),
            )

    def initialize(self) -> AppState:
        self.state.phase = "startup"
        self._emit(StartupEvent(stage="startup", message=f"loading project {self.state.project_root}"))

        def _forward(event: EngineEvent) -> None:
            self._emit(event)

        spec = load_spec_for_project(self.state.project_root, emit=_forward)
        self.state.spec = spec
        self.state.metric_name = spec.metric.name
        self.state.metric_direction = spec.metric.direction
        self.state.metric_baseline = spec.metric.baseline

        Path(spec.paths.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(spec.paths.instructions_dir).mkdir(parents=True, exist_ok=True)
        ensure_ledger(spec.paths.results_tsv)
        ensure_journal(spec.paths.journal_md, spec.name)
        ensure_metrics_log(str(self.metrics_log_path))
        ensure_memory_log(str(self.memory_log_path))

        try:
            git_state = get_state(spec.project_root) if spec.git.enabled else None
        except RuntimeError as exc:
            self.state.phase = "error"
            self._emit(ErrorEvent(message="git state unavailable", detail=str(exc)))
            raise

        self.state.anchor_commit = git_state.commit if git_state else "workspace"

        rows = read_rows(spec.paths.results_tsv)
        self.state.best_metric = best_metric(rows, spec.metric.direction)
        if self.state.best_metric is None and spec.metric.baseline is not None:
            baseline_row = LedgerRow(
                iteration=0,
                commit=self.state.anchor_commit,
                metric_name=spec.metric.name,
                metric_value=spec.metric.baseline,
                memory_gb=0.0,
                status="keep",
                description="baseline",
            )
            append_row(spec.paths.results_tsv, baseline_row)
            append_entry(
                spec.paths.journal_md,
                iteration=0,
                idea="Baseline",
                status="keep",
                metric_name=spec.metric.name,
                metric_value=spec.metric.baseline,
                commit=self.state.anchor_commit,
                log_path="",
                note="Baseline metric recorded from spec.",
            )
            append_memory_entry(
                str(self.memory_log_path),
                MemoryEntry(
                    timestamp_utc="",
                    iteration=0,
                    status="baseline",
                    metric_name=spec.metric.name,
                    metric_value=spec.metric.baseline,
                    commit=self.state.anchor_commit,
                    summary=(
                        f"baseline recorded from spec: {spec.metric.name}="
                        f"{spec.metric.baseline:.6f}"
                    ),
                    tags=infer_tags("baseline", spec.metric.name),
                ),
            )
            self.state.best_metric = spec.metric.baseline

        self._backfill_metrics_from_ledger()
        self._backfill_memory_from_ledger()

        self._load_session_events()
        self._refresh_iteration_state()
        self._refresh_telemetry_state()
        self._refresh_memory_state()

        self.state.phase = "idle"
        self._emit(
            StartupEvent(
                stage="ready",
                message=(
                    f"metric={spec.metric.name} direction={spec.metric.direction} best="
                    f"{self.state.best_metric if self.state.best_metric is not None else 'n/a'}"
                ),
            )
        )
        return self.state

    def load_history(self) -> list[IterationRecord]:
        self._refresh_iteration_state()
        return self.state.iteration_records

    def reset_history(self) -> None:
        spec = self._require_spec()
        for file_path in [
            Path(spec.paths.results_tsv),
            Path(spec.paths.journal_md),
            self.session_path,
            self.metrics_log_path,
            self.memory_log_path,
        ]:
            if file_path.exists():
                file_path.unlink()

        for dir_path in [Path(spec.paths.logs_dir), Path(spec.paths.instructions_dir)]:
            if dir_path.exists():
                shutil.rmtree(dir_path)

        ensure_ledger(spec.paths.results_tsv)
        ensure_journal(spec.paths.journal_md, spec.name)
        ensure_metrics_log(str(self.metrics_log_path))
        ensure_memory_log(str(self.memory_log_path))
        self.state.chat_history = []
        self.state.pending_action = None
        self.state.live_monitor = {}
        self.state.running_iteration = None
        self.state.running_commit = ""
        self.state.running_phase = "idle"
        self.state.token_input_total = 0
        self.state.token_output_total = 0
        self.state.token_total = 0
        self.state.last_input_tokens = None
        self.state.last_output_tokens = None
        self.state.last_total_tokens = None
        self.state.last_token_source = ""
        self._refresh_iteration_state()
        self._refresh_telemetry_state()
        self._refresh_memory_state()
        self._emit(ChatEvent(role="system", content="Reset complete: local TensorClaw history cleared."))

    def chat(self, user_text: str) -> None:
        spec = self._require_spec()

        user_text = user_text.strip()
        if not user_text:
            return

        with self._lock:
            self.state.chat_history.append(ChatMessage(role="user", content=user_text))
            has_pending = self.state.pending_action is not None
        self._append_session_event("chat", {"role": "user", "content": user_text})
        self._emit(ChatEvent(role="user", content=user_text))

        if not spec.agent.command.strip():
            reply = "Agent is not configured in this project."
            with self._lock:
                self.state.chat_history.append(ChatMessage(role="assistant", content=reply))
            self._append_session_event("chat", {"role": "assistant", "content": reply})
            self._emit(ChatEvent(role="assistant", content=reply))
            return

        if not _has_agent_credentials():
            reply = "No provider credentials found. Export OPENAI_API_KEY or ANTHROPIC_API_KEY."
            with self._lock:
                self.state.chat_history.append(ChatMessage(role="assistant", content=reply))
            self._append_session_event("chat", {"role": "assistant", "content": reply})
            self._emit(ChatEvent(role="assistant", content=reply))
            return

        if has_pending:
            self._handle_pending_input(spec=spec, user_text=user_text)
            return

        with self._lock:
            history_text = self._render_chat_history(limit=CHAT_HISTORY_LIMIT)
            best_metric = self.state.best_metric
            next_iteration = self.state.current_iteration + 1
        recent_results = summarize_recent(read_rows(spec.paths.results_tsv), limit=5)
        relevant_memory = retrieve_relevant(read_memory_entries(str(self.memory_log_path), limit=200), user_text, limit=MEMORY_CONTEXT_LIMIT)
        memory_text = format_memory_entries(relevant_memory)
        best_text = "n/a" if best_metric is None else f"{best_metric:.6f}"

        prompt = (
            f"{CHAT_DECISION_PROMPT}\n\n"
            f"Metric: {spec.metric.name} ({spec.metric.direction}), best={best_text}\n"
            f"Recent results:\n{recent_results}\n\n"
            f"Durable project memory:\n{memory_text}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Latest user message:\n{user_text}\n"
        )

        payload, fallback_reply, streamed = self._run_agent_json(
            spec=spec,
            prompt=prompt,
            stream_role="assistant",
            usage_source="chat",
        )
        if payload is None:
            assistant_reply = fallback_reply or "(No response.)"
            intent = "chat_only"
            proposal = None
        else:
            assistant_reply, intent, proposal = _parse_chat_payload(payload)

        with self._lock:
            self.state.chat_history.append(ChatMessage(role="assistant", content=assistant_reply))
        self._append_session_event("chat", {"role": "assistant", "content": assistant_reply})
        if streamed:
            self._emit(ChatStreamEvent(role="assistant", mode="end", text=assistant_reply))
        else:
            self._emit(ChatEvent(role="assistant", content=assistant_reply))

        if intent == "proposal_ready":
            if proposal is None:
                proposal = _parse_proposal_from_text(assistant_reply)
            with self._lock:
                self.state.pending_action = PendingAction(type="proposal", plan=proposal)
            self._append_session_event(
                "proposal",
                {
                    "hypothesis": proposal.hypothesis,
                    "planned_edits": proposal.planned_edits,
                    "expected_impact": proposal.expected_impact,
                    "risk": proposal.risk,
                    "direction": proposal.direction,
                },
            )
            self._emit(ProposalEvent(title=f"Plan for iteration {next_iteration}", body=proposal.to_markdown()))

    def _run_agent_json(
        self,
        *,
        spec: ResearchSpec,
        prompt: str,
        stream_role: str | None = None,
        usage_source: str = "chat",
    ) -> tuple[dict[str, Any] | None, str | None, bool]:
        command_template = _ensure_json_mode((spec.agent.proposal_command or spec.agent.command).strip())
        rendered = render_template(
            command_template,
            {
                "instruction": prompt,
                "instruction_shell": shell_escape(prompt),
                "instruction_path": "",
            },
        )
        rendered = f"{rendered} </dev/null"
        if stream_role and shutil.which("script"):
            rendered = f"script -q /dev/null sh -lc {shell_escape(rendered)}"
        stream_started = False
        stream_json_buffer = ""
        streamed_reply = ""
        usage_totals: dict[str, int] = {}

        def _stream_line(line: str) -> None:
            nonlocal stream_started, stream_json_buffer, streamed_reply
            usage = _extract_usage_from_json_line(line)
            if usage:
                _merge_usage(usage_totals, usage)
            if not stream_role:
                return
            raw_delta = _extract_stream_delta_from_json_line(line)
            if not raw_delta:
                return
            stream_json_buffer += raw_delta
            candidate_reply = _extract_assistant_reply_candidate(stream_json_buffer)
            if candidate_reply is None:
                return
            if candidate_reply.startswith(streamed_reply):
                delta = candidate_reply[len(streamed_reply) :]
            else:
                delta = candidate_reply
            streamed_reply = candidate_reply
            if not delta:
                return
            if not stream_started:
                self._emit(ChatStreamEvent(role=stream_role, mode="start"))
                stream_started = True
            self._emit(ChatStreamEvent(role=stream_role, mode="delta", text=delta))

        result = run_command(
            rendered,
            cwd=spec.project_root,
            timeout_seconds=spec.agent.timeout_seconds,
            stream_output=True,
            raw_stream_output=False,
            on_stdout_line=_stream_line,
            on_stderr_line=_stream_line,
        )

        for line in (f"{result.stdout}\n{result.stderr}").splitlines():
            usage = _extract_usage_from_json_line(line)
            if usage:
                _merge_usage(usage_totals, usage)
        if usage_totals:
            self._record_usage(usage_source, usage_totals)

        if result.timed_out:
            return None, "Agent request timed out.", stream_started
        if result.returncode != 0:
            excerpt = (result.stderr or result.stdout).strip()
            if len(excerpt) > 400:
                excerpt = excerpt[-400:]
            message = f"Agent request failed (exit {result.returncode})."
            if excerpt:
                message = f"{message}\n{excerpt}"
            return None, message, stream_started

        raw_output = f"{result.stdout}\n{result.stderr}"
        try:
            payload = _extract_payload_from_json_stream(raw_output) or _extract_json_object(raw_output)
            return payload, None, stream_started
        except ValueError:
            fallback = (result.stdout or "").strip() or (result.stderr or "").strip() or "(No response.)"
            return None, fallback, stream_started

    def _handle_pending_input(self, *, spec: ResearchSpec, user_text: str) -> None:
        with self._lock:
            pending = self.state.pending_action
        if pending is None:
            return

        with self._lock:
            history_text = self._render_chat_history(limit=CHAT_HISTORY_LIMIT)
        plan_text = pending.plan.to_markdown()
        relevant_memory = retrieve_relevant(read_memory_entries(str(self.memory_log_path), limit=200), user_text, limit=MEMORY_CONTEXT_LIMIT)
        memory_text = format_memory_entries(relevant_memory)
        prompt = (
            f"{PENDING_DECISION_PROMPT}\n\n"
            f"Pending proposal:\n{plan_text}\n\n"
            f"Durable project memory:\n{memory_text}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Latest user message:\n{user_text}\n"
        )

        payload, fallback_reply, streamed = self._run_agent_json(
            spec=spec,
            prompt=prompt,
            stream_role="assistant",
            usage_source="pending",
        )
        if payload is None:
            assistant_reply = fallback_reply or "(No response.)"
            action = "chat_only"
        else:
            assistant_reply, action = _parse_pending_payload(payload)

        with self._lock:
            self.state.chat_history.append(ChatMessage(role="assistant", content=assistant_reply))
        self._append_session_event("chat", {"role": "assistant", "content": assistant_reply})
        if streamed:
            self._emit(ChatStreamEvent(role="assistant", mode="end", text=assistant_reply))
        else:
            self._emit(ChatEvent(role="assistant", content=assistant_reply))

        if action == "approve":
            self.approve_plan()
            return
        if action == "reject":
            self.reject_plan(notify=False)
            return

    def approve_plan(self) -> None:
        with self._lock:
            pending = self.state.pending_action
            if pending is None:
                self._emit(ChatEvent(role="system", content="No pending proposal to run."))
                return
            plan = pending.plan
            self.state.pending_action = None
        self._append_session_event("proposal_cleared", {})
        direction = plan.direction.strip() or plan.hypothesis
        self.run_iteration(direction=direction, approved_plan=plan)

    def reject_plan(self, notify: bool = True) -> None:
        with self._lock:
            if not self.state.pending_action:
                self._emit(ChatEvent(role="system", content="No pending proposal to reject."))
                return
            self.state.pending_action = None
        self._append_session_event("proposal_cleared", {})
        if notify:
            self._emit(ChatEvent(role="system", content="Proposal rejected. No experiment executed."))

    def run_iteration(self, direction: str | None = None, approved_plan: ProposalPlan | None = None) -> None:
        spec = self._require_spec()
        idea = (direction or "Try one focused experiment that could improve the objective metric.").strip()
        plan_text = approved_plan.to_markdown() if approved_plan else f"Hypothesis: {idea}"

        with self._lock:
            self.state.phase = "running"
            current_best = self.state.best_metric
            anchor_commit = self.state.anchor_commit
        outcome = self._runner.run_once(
            spec=spec,
            current_best=current_best,
            anchor_commit=anchor_commit,
            idea=idea,
            approved_plan_text=plan_text,
            dry_run=self.state.dry_run,
            metrics_log_path=str(self.metrics_log_path),
            memory_log_path=str(self.memory_log_path),
            emit=self._runner_emit,
        )
        with self._lock:
            self.state.best_metric = outcome.best_metric
            self.state.anchor_commit = outcome.anchor_commit
            self.state.phase = "idle" if outcome.ok else "error"
            self.state.running_phase = "idle"
            self.state.running_iteration = None
            self.state.running_commit = ""
        self._refresh_iteration_state()
        self._refresh_telemetry_state()
        self._refresh_memory_state()

    def _render_chat_history(self, limit: int) -> str:
        if not self.state.chat_history:
            return "(no prior conversation)"
        selected = self.state.chat_history[-limit:]
        return "\n".join(f"{msg.role}: {msg.content}" for msg in selected)

    def _require_spec(self) -> ResearchSpec:
        if self.state.spec is None:
            raise RuntimeError("controller not initialized")
        return self.state.spec
