from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Callable, Optional

from .agent import AgentRunResult, run_agent_iteration
from .git_ops import commit_all, get_state, has_changes, reset_hard
from .journal import append_entry, ensure_journal
from .ledger import LedgerRow, append_row, best_metric, ensure_ledger, read_rows, summarize_recent
from .metrics import evaluate_constraint, extract_first_float, is_better
from .shell import run_command
from .spec import ResearchSpec
from .templates import render_template

READ_ONLY_TOOLS = "read,grep,find,ls"
TOOLS_FLAG_RE = re.compile(r"(--tools\s+)([^\s]+)")

ApprovalCallback = Callable[[int, str, str], bool]


def _fmt_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _pick_idea(ideas: list[str], iteration: int) -> str:
    if not ideas:
        return "Try one focused experiment that could improve the objective metric."
    index = (iteration - 1) % len(ideas)
    return ideas[index]


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


def _extract_stdout_text(log_text: str) -> str:
    marker = "\n[stdout]\n"
    marker_index = log_text.find(marker)
    if marker_index == -1:
        return ""
    start = marker_index + len(marker)
    end_marker = "\n\n[stderr]\n"
    end_index = log_text.find(end_marker, start)
    if end_index == -1:
        end_index = len(log_text)
    return log_text[start:end_index].strip()


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


def _truncate_text(text: str, max_lines: int = 24, max_chars: int = 1600) -> str:
    if not text:
        return ""
    trimmed = _trim_lines_from_end(text.strip(), max_lines=max_lines)
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 3].rstrip() + "..."


def _read_only_agent_command(command_template: str) -> str:
    if not command_template.strip():
        return command_template
    if TOOLS_FLAG_RE.search(command_template):
        return TOOLS_FLAG_RE.sub(lambda m: f"{m.group(1)}{READ_ONLY_TOOLS}", command_template, count=1)
    return f"{command_template} --tools {READ_ONLY_TOOLS}"


def _proposal_text(result: AgentRunResult) -> str:
    text = (result.stdout or "").strip()
    if text:
        return text
    log_text = _read_text_if_exists(result.log_path)
    text = _extract_stdout_text(log_text)
    if text:
        return text
    text = (result.stderr or "").strip()
    if text:
        return text
    return "No proposal text returned."


def _proposal_refs(result: AgentRunResult | None) -> str:
    if result is None:
        return ""
    parts: list[str] = []
    if result.instruction_path:
        parts.append(f"proposal_instruction={result.instruction_path}")
    if result.log_path:
        parts.append(f"proposal_log={result.log_path}")
    return "; ".join(parts)


def _has_provider_credentials() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )


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


def run_research_loop(
    spec: ResearchSpec,
    max_iterations: Optional[int] = None,
    dry_run: bool = False,
    live_output: bool = False,
    proposal_approval: Optional[ApprovalCallback] = None,
) -> int:
    Path(spec.paths.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(spec.paths.instructions_dir).mkdir(parents=True, exist_ok=True)
    ensure_ledger(spec.paths.results_tsv)
    ensure_journal(spec.paths.journal_md, spec.name)

    rows = read_rows(spec.paths.results_tsv)

    try:
        git_state = get_state(spec.project_root) if spec.git.enabled else None
    except RuntimeError as exc:
        print(f"Git state unavailable: {exc}")
        return 1

    anchor_commit = git_state.commit if git_state else "workspace"
    credentials_available = _has_provider_credentials()
    agent_enabled = bool(spec.agent.command.strip()) and credentials_available
    if not credentials_available and spec.agent.command.strip():
        print("Provider credentials were not found. Agent edits are disabled for this run.")

    current_best = best_metric(rows, spec.metric.direction)
    if current_best is None and spec.metric.baseline is not None:
        baseline_row = LedgerRow(
            iteration=0,
            commit=anchor_commit,
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
            commit=anchor_commit,
            log_path="",
            note="Baseline metric recorded from spec.",
        )
        rows = read_rows(spec.paths.results_tsv)
        current_best = spec.metric.baseline

    for setup_command in spec.commands.setup:
        if dry_run:
            print(f"Dry-run setup: {setup_command}")
            continue
        print(f"Setup: {setup_command}")
        setup_result = run_command(
            setup_command,
            cwd=spec.project_root,
            timeout_seconds=spec.loop.timeout_seconds,
            stream_output=live_output,
        )
        if setup_result.returncode != 0 or setup_result.timed_out:
            print("Setup failed; stopping.")
            return 1

    total_iterations = max_iterations if max_iterations is not None else spec.loop.max_iterations
    first_iteration = _next_iteration(rows)

    artifact_excludes = [
        spec.paths.results_tsv,
        spec.paths.journal_md,
        spec.paths.logs_dir,
        spec.paths.instructions_dir,
        *spec.git.exclude_paths,
    ]

    for offset in range(total_iterations):
        iteration = first_iteration + offset
        rows = read_rows(spec.paths.results_tsv)
        recent_results = summarize_recent(rows, limit=5)
        idea = _pick_idea(spec.ideas, iteration)

        context = {
            "iteration": iteration,
            "run_name": spec.name,
            "metric_name": spec.metric.name,
            "metric_direction": spec.metric.direction,
            "best_metric": _fmt_metric(current_best),
            "target_files_text": "\n".join(spec.target_files) if spec.target_files else "(not specified)",
            "recent_results": recent_results,
            "recent_crash_diagnostics": _summarize_recent_crashes(rows, limit=2),
            "idea": idea,
        }

        print(f"\nIteration {iteration} | best {_fmt_metric(current_best)}")
        print(f"Direction: {idea[:200]}")

        proposal_result: AgentRunResult | None = None
        proposal_text = ""
        if agent_enabled and spec.agent.proposal_enabled:
            proposal_command = spec.agent.proposal_command or _read_only_agent_command(spec.agent.command)
            proposal_result = run_agent_iteration(
                project_root=spec.project_root,
                agent_enabled=agent_enabled,
                agent_command_template=proposal_command,
                instruction_template=spec.agent.proposal_instruction_template,
                instruction_dir=spec.paths.instructions_dir,
                logs_dir=spec.paths.logs_dir,
                timeout_seconds=spec.agent.timeout_seconds,
                iteration=iteration,
                context=context,
                phase="proposal",
                save_instruction=spec.agent.save_instruction,
                stream_output=live_output and proposal_approval is None,
            )

            if proposal_result.attempted and not proposal_result.success and not spec.agent.continue_on_failure:
                append_entry(
                    spec.paths.journal_md,
                    iteration=iteration,
                    idea=idea,
                    status="proposal-failed",
                    metric_name=spec.metric.name,
                    metric_value=None,
                    commit=anchor_commit,
                    log_path=proposal_result.log_path,
                    instruction_path=proposal_result.instruction_path,
                    note=proposal_result.error,
                )
                print("Proposal step failed and continue_on_failure=false; stopping.")
                return 1

            proposal_text = _proposal_text(proposal_result)
            if proposal_approval is None:
                proposal_excerpt = _truncate_text(proposal_text, max_lines=20, max_chars=1400)
                print("Proposal:")
                if proposal_excerpt:
                    for line in proposal_excerpt.splitlines():
                        print(f"  {line}")
                else:
                    print("  (empty proposal)")

            if proposal_approval is not None:
                approved = proposal_approval(iteration, idea, proposal_text)
                if not approved:
                    if spec.git.enabled and spec.git.discard_strategy == "hard-reset":
                        reset_hard(spec.project_root, anchor_commit)
                    row = LedgerRow(
                        iteration=iteration,
                        commit=anchor_commit,
                        metric_name=spec.metric.name,
                        metric_value=current_best if current_best is not None else 0.0,
                        memory_gb=0.0,
                        status="proposal-rejected",
                        description=idea,
                        constraints="",
                        log_path=proposal_result.log_path,
                    )
                    append_row(spec.paths.results_tsv, row)
                    append_entry(
                        spec.paths.journal_md,
                        iteration=iteration,
                        idea=idea,
                        status="proposal-rejected",
                        metric_name=spec.metric.name,
                        metric_value=current_best,
                        commit=anchor_commit,
                        log_path=proposal_result.log_path,
                        instruction_path=proposal_result.instruction_path,
                        note="Proposal rejected by user.",
                    )
                    print("Plan rejected; skipped apply/experiment.")
                    continue

        apply_context = dict(context)
        apply_context["approved_plan"] = proposal_text if proposal_text else f"Hypothesis only: {idea}"
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
            stream_output=live_output,
        )

        if agent_result.attempted and not agent_result.success and not spec.agent.continue_on_failure:
            note = agent_result.error
            proposal_note = _proposal_refs(proposal_result)
            if proposal_note:
                note = f"{note}; {proposal_note}" if note else proposal_note
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
                note=note,
            )
            print("Agent step failed and continue_on_failure=false; stopping.")
            return 1

        proposal_note = _proposal_refs(proposal_result)
        tested_commit = anchor_commit
        if spec.git.enabled:
            current_state = get_state(spec.project_root)
            tested_commit = current_state.commit

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
                note="; ".join(
                    part
                    for part in [
                        "Dry-run: reverted tracked workspace changes.",
                        proposal_note,
                    ]
                    if part
                ),
            )
            continue

        if spec.git.enabled and spec.git.auto_commit:
            commit_message = render_template(spec.git.commit_message_template, context)
            if has_changes(spec.project_root):
                try:
                    committed = commit_all(
                        spec.project_root,
                        message=commit_message,
                        exclude_paths=artifact_excludes,
                    )
                except RuntimeError as exc:
                    print(f"Commit failed: {exc}")
                    return 1

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
                        note="; ".join(
                            part
                            for part in [
                                "No code changes staged for commit.",
                                proposal_note,
                            ]
                            if part
                        ),
                    )
                    continue
                tested_commit = committed

        run_log_path = str(Path(spec.paths.logs_dir) / f"run_{iteration:04d}.log")
        run_result = run_command(
            spec.commands.experiment,
            cwd=spec.project_root,
            timeout_seconds=spec.loop.timeout_seconds,
            log_path=run_log_path,
            stream_output=live_output,
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
            note="; ".join(
                note_parts
                + ([constraints_text] if constraints_text else [])
                + ([proposal_note] if proposal_note else [])
            ),
        )

        print(
            f"Result: {status} | metric {_fmt_metric(metric_value)} | best {_fmt_metric(current_best)}"
        )

    return 0
