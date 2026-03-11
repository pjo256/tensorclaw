from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..agent import AgentRunResult, run_agent_iteration
from ..git_ops import commit_all, get_state, has_changes, reset_hard
from ..journal import append_entry
from ..ledger import LedgerRow, append_row, read_rows, summarize_recent
from ..memory import MemoryEntry, append_memory_entry, infer_tags
from ..metrics import evaluate_constraint, extract_first_float, is_better
from ..shell import run_command
from ..spec import ResearchSpec
from ..telemetry import MetricSample, append_metric_sample
from ..templates import render_template
from .events import DecisionEvent, ErrorEvent, IterationEvent, MetricEvent, OutputLineEvent, TelemetryEvent


Emitter = Callable[[object], None] | None

READ_ONLY_TOOLS = "read,grep,find,ls"
TOOLS_FLAG_RE = re.compile(r"(--tools\s+)([^\s]+)")
STEP_METRIC_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\((?P<progress_pct>[-+]?\d*\.?\d+)%\)\s+\|\s+"
    r"loss:\s*(?P<train_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+\|\s+"
    r"lrm:\s*(?P<lrm>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+\|\s+"
    r"dt:\s*(?P<dt_ms>\d+)ms\s+\|\s+"
    r"tok/sec:\s*(?P<tok_per_sec>[0-9][0-9,]*(?:\.\d+)?)\s+\|\s+"
    r"epoch:\s*(?P<epoch>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+\|\s+"
    r"remaining:\s*(?P<remaining_s>\d+)s",
    re.IGNORECASE,
)
KV_METRIC_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9_/\-%\.]*)\s*:\s*"
    r"(?P<value>[-+]?[0-9][0-9,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


@dataclass(slots=True)
class RunOutcome:
    ok: bool
    iteration: int
    status: str
    metric_value: float | None
    best_metric: float | None
    anchor_commit: str


def _emit(emit: Emitter, event: object) -> None:
    if emit:
        emit(event)


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _read_text_if_exists(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _extract_stderr_text(log_text: str) -> str:
    marker = "\n[stderr]\n"
    marker_index = log_text.rfind(marker)
    if marker_index == -1:
        return log_text.strip()
    return log_text[marker_index + len(marker) :].strip()


def _trim_lines_from_end(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _extract_failure_excerpt(log_path: str, max_lines: int = 20, max_chars: int = 1200) -> str:
    raw_text = _read_text_if_exists(log_path)
    if not raw_text:
        return "No log excerpt available."

    stderr_text = _extract_stderr_text(raw_text)
    if not stderr_text:
        return "stderr was empty."

    traceback_marker = "Traceback (most recent call last):"
    traceback_index = stderr_text.rfind(traceback_marker)
    excerpt = stderr_text[traceback_index:] if traceback_index != -1 else stderr_text
    excerpt = _trim_lines_from_end(excerpt, max_lines=max_lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
    return excerpt.strip()


def _summarize_recent_crashes(rows: list[dict[str, str]], limit: int = 2) -> str:
    crash_rows = [row for row in rows if row.get("status") == "crash"]
    if not crash_rows:
        return "No recent crash diagnostics."

    selected = crash_rows[-limit:]
    blocks: list[str] = []
    for row in selected:
        iteration = row.get("iteration", "?")
        log_path = row.get("log_path", "")
        excerpt = _extract_failure_excerpt(log_path)
        blocks.append(f"iter {iteration} crash excerpt:\n{excerpt}")
    return "\n\n".join(blocks)


def _has_provider_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _normalize_metric_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("/", "_per_")
    normalized = normalized.replace("%", "_pct")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_live_metrics(line: str) -> dict[str, float]:
    text = line.strip()
    if not text:
        return {}

    step_match = STEP_METRIC_RE.search(text)
    if step_match:
        metrics: dict[str, float] = {}
        for key, value in step_match.groupdict().items():
            parsed = _parse_number(value)
            if parsed is None:
                continue
            metrics[key] = parsed
        return metrics

    kv_match = KV_METRIC_RE.match(text)
    if not kv_match:
        return {}
    metric_name = _normalize_metric_name(kv_match.group("name"))
    if not metric_name:
        return {}
    parsed_value = _parse_number(kv_match.group("value"))
    if parsed_value is None:
        return {}
    return {metric_name: parsed_value}


def _constraint_summary(spec: ResearchSpec, text: str) -> tuple[bool, str]:
    if not spec.constraints:
        return True, ""

    parts: list[str] = []
    passed_all = True
    for constraint in spec.constraints:
        value = extract_first_float(text, constraint.pattern)
        passed = evaluate_constraint(value, constraint.op, constraint.threshold)
        passed_all = passed_all and passed
        value_text = "n/a" if value is None else f"{value:.6f}"
        verdict = "pass" if passed else "fail"
        parts.append(f"{constraint.name}={value_text} {constraint.op} {constraint.threshold:g} ({verdict})")

    return passed_all, "; ".join(parts)


def _next_iteration(rows: list[dict[str, str]]) -> int:
    current_max = 0
    for row in rows:
        try:
            current_max = max(current_max, int(row.get("iteration", "0")))
        except ValueError:
            continue
    return current_max + 1


def _emit_run_output(emit: Emitter, phase: str, stream: str) -> Callable[[str], None]:
    def _inner(line: str) -> None:
        _emit(emit, OutputLineEvent(phase=phase, stream="stderr" if stream == "stderr" else "stdout", line=line.rstrip("\n")))

    return _inner


class IterationRunner:
    def run_once(
        self,
        *,
        spec: ResearchSpec,
        current_best: float | None,
        anchor_commit: str,
        idea: str,
        approved_plan_text: str,
        dry_run: bool,
        metrics_log_path: str | None = None,
        memory_log_path: str | None = None,
        emit: Emitter = None,
    ) -> RunOutcome:
        rows = read_rows(spec.paths.results_tsv)
        iteration = _next_iteration(rows)
        _emit(emit, IterationEvent(iteration=iteration, status="start", message=f"iteration {iteration} started", commit=anchor_commit))

        def _write_memory(status: str, *, metric_value: float | None, commit: str, note: str) -> None:
            if not memory_log_path:
                return
            summary = (
                f"idea={idea}; status={status}; {spec.metric.name}="
                f"{'n/a' if metric_value is None else f'{metric_value:.6f}'}; {note}"
            ).strip()
            entry = MemoryEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                iteration=iteration,
                status=status,
                metric_name=spec.metric.name,
                metric_value=metric_value,
                commit=commit,
                summary=summary,
                tags=infer_tags(idea, note, approved_plan_text, summary),
            )
            append_memory_entry(memory_log_path, entry)

        for setup_command in spec.commands.setup:
            _emit(emit, OutputLineEvent(phase="setup", stream="meta", line=f"$ {setup_command}"))
            setup_result = run_command(
                setup_command,
                cwd=spec.project_root,
                timeout_seconds=spec.loop.timeout_seconds,
                stream_output=True,
                raw_stream_output=False,
                on_stdout_line=_emit_run_output(emit, "setup", "stdout"),
                on_stderr_line=_emit_run_output(emit, "setup", "stderr"),
            )
            if setup_result.returncode != 0 or setup_result.timed_out:
                msg = "setup failed"
                _emit(emit, ErrorEvent(message=msg, detail=f"return={setup_result.returncode} timed_out={setup_result.timed_out}"))
                _write_memory(
                    "setup-failed",
                    metric_value=current_best,
                    commit=anchor_commit,
                    note=f"setup command failed: {setup_command}",
                )
                return RunOutcome(
                    ok=False,
                    iteration=iteration,
                    status="setup-failed",
                    metric_value=None,
                    best_metric=current_best,
                    anchor_commit=anchor_commit,
                )

        credentials_available = _has_provider_credentials()
        agent_enabled = bool(spec.agent.command.strip()) and credentials_available
        if not credentials_available and spec.agent.command.strip():
            _emit(emit, OutputLineEvent(phase="apply", stream="meta", line="provider credentials not found; agent edits disabled"))

        apply_context = {
            "iteration": iteration,
            "run_name": spec.name,
            "metric_name": spec.metric.name,
            "metric_direction": spec.metric.direction,
            "best_metric": _fmt_metric(current_best),
            "target_files_text": "\n".join(spec.target_files) if spec.target_files else "(not specified)",
            "recent_results": summarize_recent(rows, limit=5),
            "recent_crash_diagnostics": _summarize_recent_crashes(rows, limit=2),
            "idea": idea,
            "approved_plan": approved_plan_text if approved_plan_text else f"Hypothesis only: {idea}",
        }

        agent_result = run_agent_iteration(
            project_root=spec.project_root,
            agent_enabled=agent_enabled,
            agent_command_template=spec.agent.command,
            instruction_template=spec.agent.instruction_template,
            instruction_dir=spec.paths.instructions_dir,
            logs_dir=spec.paths.logs_dir,
            timeout_seconds=spec.agent.timeout_seconds,
            iteration=iteration,
            context=apply_context,
            phase="apply",
            save_instruction=spec.agent.save_instruction,
            stream_output=True,
            raw_stream_output=False,
            on_stdout_line=_emit_run_output(emit, "apply", "stdout"),
            on_stderr_line=_emit_run_output(emit, "apply", "stderr"),
        )

        if agent_result.attempted and not agent_result.success and not spec.agent.continue_on_failure:
            append_entry(
                spec.paths.journal_md,
                iteration=iteration,
                idea=idea,
                status="agent-failed",
                metric_name=spec.metric.name,
                metric_value=None,
                commit=anchor_commit,
                log_path=agent_result.log_path,
                instruction_path=agent_result.instruction_path,
                note=agent_result.error,
            )
            _emit(emit, ErrorEvent(message="agent step failed", detail=agent_result.error))
            _write_memory(
                "agent-failed",
                metric_value=current_best,
                commit=anchor_commit,
                note=agent_result.error or "agent failed",
            )
            return RunOutcome(
                ok=False,
                iteration=iteration,
                status="agent-failed",
                metric_value=None,
                best_metric=current_best,
                anchor_commit=anchor_commit,
            )

        tested_commit = anchor_commit
        if spec.git.enabled:
            current_state = get_state(spec.project_root)
            tested_commit = current_state.commit

        artifact_excludes = [
            spec.paths.results_tsv,
            spec.paths.journal_md,
            spec.paths.logs_dir,
            spec.paths.instructions_dir,
            *spec.git.exclude_paths,
        ]

        if dry_run:
            if spec.git.enabled and spec.git.discard_strategy == "hard-reset":
                reset_hard(spec.project_root, tested_commit)
            row = LedgerRow(
                iteration=iteration,
                commit=tested_commit,
                metric_name=spec.metric.name,
                metric_value=current_best if current_best is not None else 0.0,
                memory_gb=0.0,
                status="dry-run",
                description=idea,
                constraints="",
                log_path="",
            )
            append_row(spec.paths.results_tsv, row)
            append_entry(
                spec.paths.journal_md,
                iteration=iteration,
                idea=idea,
                status="dry-run",
                metric_name=spec.metric.name,
                metric_value=current_best,
                commit=tested_commit,
                log_path="",
                instruction_path=agent_result.instruction_path,
                note="Dry-run: reverted tracked workspace changes.",
            )
            _emit(emit, DecisionEvent(iteration=iteration, status="dry-run", metric=current_best, best=current_best, commit=tested_commit))
            _write_memory(
                "dry-run",
                metric_value=current_best,
                commit=tested_commit,
                note="dry-run executed, workspace reverted",
            )
            return RunOutcome(
                ok=True,
                iteration=iteration,
                status="dry-run",
                metric_value=current_best,
                best_metric=current_best,
                anchor_commit=anchor_commit,
            )

        if spec.git.enabled and spec.git.auto_commit:
            commit_message = render_template(spec.git.commit_message_template, apply_context)
            if has_changes(spec.project_root):
                try:
                    committed = commit_all(spec.project_root, message=commit_message, exclude_paths=artifact_excludes)
                except RuntimeError as exc:
                    _emit(emit, ErrorEvent(message="commit failed", detail=str(exc)))
                    _write_memory(
                        "commit-failed",
                        metric_value=current_best,
                        commit=anchor_commit,
                        note=str(exc),
                    )
                    return RunOutcome(
                        ok=False,
                        iteration=iteration,
                        status="commit-failed",
                        metric_value=None,
                        best_metric=current_best,
                        anchor_commit=anchor_commit,
                    )

                if committed is None:
                    row = LedgerRow(
                        iteration=iteration,
                        commit=tested_commit,
                        metric_name=spec.metric.name,
                        metric_value=current_best if current_best is not None else 0.0,
                        memory_gb=0.0,
                        status="no-change",
                        description=idea,
                        constraints="",
                        log_path=agent_result.log_path,
                    )
                    append_row(spec.paths.results_tsv, row)
                    append_entry(
                        spec.paths.journal_md,
                        iteration=iteration,
                        idea=idea,
                        status="no-change",
                        metric_name=spec.metric.name,
                        metric_value=current_best,
                        commit=tested_commit,
                        log_path=agent_result.log_path,
                        instruction_path=agent_result.instruction_path,
                        note="No code changes staged for commit.",
                    )
                    _emit(emit, DecisionEvent(iteration=iteration, status="no-change", metric=current_best, best=current_best, commit=tested_commit))
                    _write_memory(
                        "no-change",
                        metric_value=current_best,
                        commit=tested_commit,
                        note="agent produced no staged changes",
                    )
                    return RunOutcome(
                        ok=True,
                        iteration=iteration,
                        status="no-change",
                        metric_value=current_best,
                        best_metric=current_best,
                        anchor_commit=anchor_commit,
                    )
                tested_commit = committed

        run_log_path = str(Path(spec.paths.logs_dir) / f"run_{iteration:04d}.log")
        _emit(
            emit,
            IterationEvent(
                iteration=iteration,
                status="experiment",
                message=f"iteration {iteration} running on commit {tested_commit}",
                commit=tested_commit,
            ),
        )
        _emit(emit, OutputLineEvent(phase="experiment", stream="meta", line=f"$ {spec.commands.experiment}"))

        def _emit_experiment_line(stream: str) -> Callable[[str], None]:
            def _inner(line: str) -> None:
                stripped = line.rstrip("\n")
                _emit(
                    emit,
                    OutputLineEvent(
                        phase="experiment",
                        stream="stderr" if stream == "stderr" else "stdout",
                        line=stripped,
                    ),
                )
                parsed = _parse_live_metrics(stripped)
                if not parsed:
                    return
                _emit(emit, TelemetryEvent(iteration=iteration, phase="experiment", metrics=parsed))
                if not metrics_log_path:
                    return
                append_metric_sample(
                    metrics_log_path,
                    MetricSample(
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        iteration=iteration,
                        phase="experiment",
                        metrics=parsed,
                        raw_line=stripped,
                    ),
                )

            return _inner

        run_result = run_command(
            spec.commands.experiment,
            cwd=spec.project_root,
            timeout_seconds=spec.loop.timeout_seconds,
            log_path=run_log_path,
            stream_output=True,
            raw_stream_output=False,
            on_stdout_line=_emit_experiment_line("stdout"),
            on_stderr_line=_emit_experiment_line("stderr"),
        )

        output_text = f"{run_result.stdout}\n{run_result.stderr}"
        metric_value = extract_first_float(output_text, spec.metric.pattern)

        memory_gb = 0.0
        if spec.memory and spec.memory.pattern:
            raw_memory = extract_first_float(output_text, spec.memory.pattern)
            if raw_memory is not None:
                memory_gb = raw_memory * spec.memory.scale_to_gb

        constraints_pass, constraints_text = _constraint_summary(spec, output_text)

        is_crash = run_result.timed_out or run_result.returncode != 0 or metric_value is None
        keep = False
        status = "crash"
        recorded_metric = 0.0
        note_parts: list[str] = []

        if run_result.timed_out:
            note_parts.append("experiment timed out")
        elif run_result.returncode != 0:
            note_parts.append(f"experiment exited {run_result.returncode}")

        if is_crash:
            status = "crash"
        else:
            recorded_metric = metric_value if metric_value is not None else 0.0
            improved = is_better(
                candidate=recorded_metric,
                best=current_best,
                direction=spec.metric.direction,
                min_delta=spec.metric.min_delta,
            )
            keep = improved and constraints_pass
            status = "keep" if keep else "discard"

            if not improved:
                note_parts.append("metric did not improve")
            if not constraints_pass:
                note_parts.append("constraints failed")

        if keep:
            current_best = recorded_metric
            if spec.git.enabled:
                anchor_commit = tested_commit
        elif spec.git.enabled and spec.git.discard_strategy == "hard-reset":
            if status == "discard" or (status == "crash" and spec.git.revert_on_crash):
                reset_hard(spec.project_root, anchor_commit)

        ledger_row = LedgerRow(
            iteration=iteration,
            commit=tested_commit,
            metric_name=spec.metric.name,
            metric_value=recorded_metric,
            memory_gb=memory_gb,
            status=status,
            description=idea,
            constraints=constraints_text,
            log_path=run_log_path,
        )
        append_row(spec.paths.results_tsv, ledger_row)

        append_entry(
            spec.paths.journal_md,
            iteration=iteration,
            idea=idea,
            status=status,
            metric_name=spec.metric.name,
            metric_value=metric_value,
            commit=tested_commit,
            log_path=run_log_path,
            instruction_path=agent_result.instruction_path,
            note="; ".join(note_parts + ([constraints_text] if constraints_text else [])),
        )

        final_metrics: dict[str, float] = {}
        if metric_value is not None:
            final_metrics[_normalize_metric_name(spec.metric.name)] = metric_value
        if memory_gb > 0:
            final_metrics["memory_gb"] = memory_gb
        if final_metrics:
            _emit(emit, TelemetryEvent(iteration=iteration, phase="final", metrics=final_metrics))
            if metrics_log_path:
                append_metric_sample(
                    metrics_log_path,
                    MetricSample(
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        iteration=iteration,
                        phase="final",
                        metrics=final_metrics,
                        raw_line=f"status={status}",
                    ),
                )

        crash_excerpt = ""
        if status == "crash":
            crash_excerpt = _extract_failure_excerpt(run_log_path)
        memory_note_parts = note_parts[:]
        if constraints_text:
            memory_note_parts.append(constraints_text)
        if crash_excerpt:
            memory_note_parts.append(crash_excerpt)
        _write_memory(
            status,
            metric_value=metric_value,
            commit=tested_commit,
            note="; ".join(part for part in memory_note_parts if part) or "iteration complete",
        )

        _emit(
            emit,
            MetricEvent(
                metric_name=spec.metric.name,
                direction=spec.metric.direction,
                value=metric_value,
                best=current_best,
                baseline=spec.metric.baseline,
            ),
        )
        _emit(
            emit,
            DecisionEvent(
                iteration=iteration,
                status=status,
                metric=metric_value,
                best=current_best,
                commit=tested_commit,
            ),
        )
        _emit(
            emit,
            IterationEvent(
                iteration=iteration,
                status=status,
                message=f"iter={iteration} status={status} metric={_fmt_metric(metric_value)} best={_fmt_metric(current_best)}",
            ),
        )

        return RunOutcome(
            ok=True,
            iteration=iteration,
            status=status,
            metric_value=metric_value,
            best_metric=current_best,
            anchor_commit=anchor_commit,
        )
