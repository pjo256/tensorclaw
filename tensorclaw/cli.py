from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .journal import ensure_journal
from .ledger import best_metric, ensure_ledger, read_rows, summarize_recent
from .loop import run_research_loop
from .metrics import extract_first_float
from .spec import ResearchSpec, load_spec
from .shell import run_command
from .templates import render_template, shell_escape

FLOAT_GROUP = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
DEFAULT_METRIC_PATTERN = rf"^val_bpb:\s*({FLOAT_GROUP})"
DEFAULT_MEMORY_PATTERN = rf"^peak_vram_mb:\s*({FLOAT_GROUP})"
DEFAULT_OPENAI_AGENT_COMMAND = (
    "pi -p --provider openai --model gpt-5.3-codex "
    "--tools read,bash,edit,write {instruction_shell}"
)
DEFAULT_ANTHROPIC_AGENT_COMMAND = (
    "pi -p --provider anthropic --model claude-sonnet-4 "
    "--tools read,bash,edit,write {instruction_shell}"
)
GENERATED_SPEC_REL_PATH = Path(".tensorclaw/spec.generated.yaml")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
KNOWN_METRICS: dict[str, tuple[str, str]] = {
    "val_bpb": ("minimize", rf"^val_bpb:\s*({FLOAT_GROUP})"),
    "val_loss": ("minimize", rf"^val_loss:\s*({FLOAT_GROUP})"),
    "eval_loss": ("minimize", rf"^eval_loss:\s*({FLOAT_GROUP})"),
    "loss": ("minimize", rf"^loss:\s*({FLOAT_GROUP})"),
    "val_accuracy": ("maximize", rf"^val_accuracy:\s*({FLOAT_GROUP})"),
    "accuracy": ("maximize", rf"^accuracy:\s*({FLOAT_GROUP})"),
    "val_acc": ("maximize", rf"^val_acc:\s*({FLOAT_GROUP})"),
    "acc": ("maximize", rf"^acc:\s*({FLOAT_GROUP})"),
}
READ_ONLY_TOOLS = "read,grep,find,ls"
AGENT_TOOLS_FLAG_RE = re.compile(r"(--tools\s+)([^\s]+)")
AGENT_MODE_FLAG_RE = re.compile(r"(--mode\s+)([^\s]+)")
DISCOVERY_TIMEOUT_SECONDS = 120
COMMAND_PROBE_TIMEOUT_SECONDS = 20
DISCOVERY_ATTEMPTS = 2
DISCOVERY_PROMPT = """You are configuring TensorClaw for this repository.

Read the codebase and return ONLY a JSON object (no prose) with:
- target_files: list of key editable files for training loop changes (paths relative to repo root)
- experiment_command: one command string runnable from repo root that launches the main train/eval run
- metric_name: scalar metric to optimize (e.g. val_bpb, val_loss, accuracy)
- metric_direction: either "minimize" or "maximize"
- metric_pattern: regex with one capture group for the metric value in run logs

Constraints:
- Use a command that matches the normal experiment flow in this repo.
- Prefer objective validation metrics over training loss.
- Paths must be repo-relative.
- Return valid JSON only.
"""

INTERACTIVE_HELP = """Commands:
  Enter              Run one iteration
  <text>             Chat with the agent
  /run N             Run N iterations
  /reset             Clear local TensorClaw history for this project
  /status            Show all iterations and best result
  /tail [N]          Tail latest experiment log (default N=40)
"""

CHAT_HISTORY_LIMIT = 8
CHAT_ACTIONS = {"none", "run_iteration", "run_iterations", "status", "tail"}
CHAT_DECISION_PROMPT = """You are TensorClaw's interactive research copilot.

Given the project context and recent conversation, return ONLY a JSON object with:
{
  "assistant_reply": "short natural-language reply for the user",
  "action": {
    "type": "none | run_iteration | run_iterations | status | tail",
    "direction": "optional direction text for run_iteration",
    "count": 1,
    "tail_lines": 40
  }
}

Rules:
- Keep assistant_reply concise (1-3 sentences).
- Do NOT mention slash commands or keyboard instructions.
- If user is asking a question, answer it and set action.type="none".
- If user explicitly asks to run/train now, set action.type to run_iteration or run_iterations.
- If user confirms a previous suggestion ("yes", "do it", "run that"), infer the intended run and set run_iteration with a concrete direction.
- For action.type other than run_iterations or tail, count/tail_lines may be omitted.
- Return valid JSON only.
"""


@dataclass
class ChatDecision:
    assistant_reply: str
    action_type: str = "none"
    direction: str = ""
    count: int = 1
    tail_lines: int = 40


def _load_yaml_module():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to initialize TensorClaw. Install dependencies first (pip install -e .)."
        ) from exc
    return yaml


def _resolve_generated_spec_path(project_root: str) -> Path:
    root = Path(project_root).expanduser().resolve()
    return root / GENERATED_SPEC_REL_PATH


def _normalize_target_file(project_root: Path, path_like: str) -> str:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _discover_agent_command(explicit: str | None) -> str:
    if explicit:
        return explicit

    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    if has_anthropic and not has_openai:
        return DEFAULT_ANTHROPIC_AGENT_COMMAND
    return DEFAULT_OPENAI_AGENT_COMMAND


def _derive_proposal_command(agent_command: str) -> str:
    if AGENT_TOOLS_FLAG_RE.search(agent_command):
        return AGENT_TOOLS_FLAG_RE.sub(
            lambda match: f"{match.group(1)}{READ_ONLY_TOOLS}",
            agent_command,
            count=1,
        )
    return f"{agent_command} --tools {READ_ONLY_TOOLS}"


def _has_agent_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Agent discovery returned empty output.")

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

    raise ValueError("Agent discovery did not return a valid JSON object.")


def _ensure_json_mode(agent_command: str) -> str:
    if AGENT_MODE_FLAG_RE.search(agent_command):
        return AGENT_MODE_FLAG_RE.sub(lambda match: f"{match.group(1)}json", agent_command, count=1)
    return f"{agent_command} --mode json"


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
        line = line.strip()
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


def _read_text_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _validated_target_files(project_root: Path, value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for item in items:
        if not item:
            continue
        rel = _normalize_target_file(project_root, str(item))
        candidate = project_root / rel
        if candidate.is_file():
            normalized.append(rel)
    seen: set[str] = set()
    deduped: list[str] = []
    for rel in normalized:
        if rel not in seen:
            seen.add(rel)
            deduped.append(rel)
    return deduped


def _tail_text_excerpt(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _run_agent_discovery(
    project_root: Path,
    agent_command: str,
    feedback: str = "",
) -> dict[str, Any]:
    if not _has_agent_credentials():
        raise ValueError("No API key found. Export OPENAI_API_KEY or ANTHROPIC_API_KEY before running tensorclaw.")

    discovery_command_template = _ensure_json_mode(_derive_proposal_command(agent_command))
    prompt = DISCOVERY_PROMPT
    if feedback:
        prompt = (
            f"{DISCOVERY_PROMPT}\n\n"
            "Previous discovery failed validation. Fix it and return corrected JSON only.\n"
            f"{feedback}\n"
        )
    rendered = render_template(
        discovery_command_template,
        {
            "instruction": prompt,
            "instruction_shell": shell_escape(prompt),
            "instruction_path": "",
        },
    )
    # Prevent pi from consuming stdin intended for tensorclaw's interactive prompt.
    rendered = f"{rendered} </dev/null"
    log_path = project_root / ".tensorclaw" / "logs" / "agent_discovery.log"
    result = run_command(
        rendered,
        cwd=str(project_root),
        timeout_seconds=DISCOVERY_TIMEOUT_SECONDS,
        log_path=str(log_path),
        stream_output=False,
    )
    if result.timed_out:
        raise ValueError("Agent discovery timed out.")
    if result.returncode != 0:
        excerpt = _tail_text_excerpt(result.stderr or result.stdout)
        raise ValueError(f"Agent discovery failed (exit {result.returncode}).\n{excerpt}")

    raw_output = f"{result.stdout}\n{result.stderr}"
    payload = _extract_payload_from_json_stream(raw_output) or _extract_json_object(raw_output)
    return payload


def _normalize_experiment_command(command: str) -> str:
    normalized = command.strip()
    if not normalized:
        return normalized
    py = shell_escape(sys.executable)
    normalized = re.sub(r"^\s*python3?\b", py, normalized, count=1)
    return normalized


def _validate_experiment_command(project_root: Path, command: str) -> None:
    probe = run_command(
        command,
        cwd=str(project_root),
        timeout_seconds=COMMAND_PROBE_TIMEOUT_SECONDS,
        stream_output=False,
    )
    if probe.timed_out:
        return
    if probe.returncode == 0:
        return
    excerpt = _tail_text_excerpt(probe.stderr or probe.stdout)
    raise ValueError(f"Discovered experiment command failed validation (exit {probe.returncode}).\n{excerpt}")


def _metric_defaults(metric_name: str) -> tuple[str, str]:
    if metric_name in KNOWN_METRICS:
        return KNOWN_METRICS[metric_name]
    return "minimize", rf"^{re.escape(metric_name)}:\s*({FLOAT_GROUP})"


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_baseline_from_results_tsv(project_root: Path, metric_name: str, direction: str) -> float | None:
    tsv_path = project_root / ".tensorclaw" / "results.tsv"
    if not tsv_path.is_file():
        return None

    keep_values: list[float] = []
    fallback_values: list[float] = []

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("metric_name") != metric_name:
                continue
            metric_value = _safe_float(row.get("metric_value"))
            if metric_value is None:
                continue
            fallback_values.append(metric_value)
            if row.get("status") == "keep":
                keep_values.append(metric_value)

    values = keep_values or fallback_values
    if not values:
        return None
    if direction == "maximize":
        return max(values)
    return min(values)


def _infer_baseline_from_log(project_root: Path, metric_pattern: str) -> float | None:
    baseline_log = _read_text_if_exists(project_root / "run_baseline.log")
    if not baseline_log:
        return None
    return extract_first_float(baseline_log, metric_pattern)


def _write_generated_spec(path: Path, raw: dict[str, Any]) -> None:
    yaml = _load_yaml_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def _default_init_args(project_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        project_root=str(project_root),
        name=project_root.name,
        target_file=[],
        experiment_command=None,
        agent_command=None,
        metric_name=None,
        metric_direction=None,
        metric_pattern=None,
        metric_baseline=None,
        metric_min_delta=0.0001,
        memory_pattern=DEFAULT_MEMORY_PATTERN,
        max_iterations=20,
        timeout_seconds=1200,
        agent_timeout_seconds=600,
    )


def _write_spec_with_summary(spec_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw = _build_generated_spec(args)
    _write_generated_spec(spec_path, raw)
    return raw


def _build_generated_spec(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).expanduser().resolve()
    agent_command = _discover_agent_command(args.agent_command)
    target_files: list[str] = []
    experiment_command = ""
    metric_name = "val_bpb"
    metric_direction = "minimize"
    metric_pattern = DEFAULT_METRIC_PATTERN
    last_error = ""

    for _ in range(1, DISCOVERY_ATTEMPTS + 1):
        discovery = _run_agent_discovery(
            project_root,
            agent_command,
            feedback=last_error,
        )

        target_files = _validated_target_files(project_root, discovery.get("target_files"))
        if not target_files and args.target_file:
            target_files = _validated_target_files(project_root, args.target_file)
        if not target_files:
            last_error = "target_files were missing or invalid."
            continue

        experiment_command = _normalize_experiment_command(str(discovery.get("experiment_command", "")).strip())
        if not experiment_command:
            last_error = "experiment_command was missing."
            continue

        metric_name = str(discovery.get("metric_name", "")).strip() or "val_bpb"
        metric_direction_default, metric_pattern_default = _metric_defaults(metric_name)
        metric_direction = str(discovery.get("metric_direction", "")).strip().lower() or metric_direction_default
        if metric_direction not in {"minimize", "maximize"}:
            metric_direction = metric_direction_default

        metric_pattern = str(discovery.get("metric_pattern", "")).strip() or metric_pattern_default
        try:
            re.compile(metric_pattern, re.MULTILINE)
        except re.error as exc:
            last_error = f"metric_pattern was invalid: {exc}"
            continue

        try:
            _validate_experiment_command(project_root, experiment_command)
        except ValueError as exc:
            last_error = str(exc)
            continue
        break
    else:
        detail = f"\nlast_error: {last_error}" if last_error else ""
        raise ValueError(f"Agent discovery failed validation after {DISCOVERY_ATTEMPTS} attempts.{detail}")

    baseline = args.metric_baseline
    if baseline is None:
        baseline = _infer_baseline_from_results_tsv(project_root, metric_name, metric_direction)
    if baseline is None:
        baseline = _infer_baseline_from_log(project_root, metric_pattern)

    metric: dict[str, Any] = {
        "name": metric_name,
        "direction": metric_direction,
        "pattern": metric_pattern,
        "min_delta": args.metric_min_delta,
    }
    if baseline is not None:
        metric["baseline"] = baseline

    agent_enabled = _has_agent_credentials()
    proposal_command = _derive_proposal_command(agent_command)

    return {
        "version": 1,
        "name": args.name,
        "project_root": str(project_root),
        "target_files": target_files,
        "commands": {
            "experiment": experiment_command,
        },
        "metric": metric,
        "memory": {
            "pattern": args.memory_pattern,
            "scale_to_gb": 0.0009765625,
        },
        "loop": {
            "max_iterations": args.max_iterations,
            "timeout_seconds": args.timeout_seconds,
        },
        "agent": {
            "enabled": agent_enabled,
            "command": agent_command,
            "proposal_enabled": True,
            "proposal_command": proposal_command,
            "timeout_seconds": args.agent_timeout_seconds,
            "continue_on_failure": False,
            "save_instruction": True,
        },
        "git": {
            "enabled": True,
            "auto_commit": True,
            "discard_strategy": "hard-reset",
            "revert_on_crash": True,
            "commit_message_template": "tensorclaw iter {iteration}: {idea}",
            "exclude_paths": [".tensorclaw", "run.log", "run_baseline.log"],
        },
        "paths": {
            "results_tsv": ".tensorclaw/results.tsv",
            "journal_md": ".tensorclaw/journal.md",
            "logs_dir": ".tensorclaw/logs",
            "instructions_dir": ".tensorclaw/instructions",
        },
        "ideas": [
            "Try one minimal change that can improve the main metric.",
            "Prefer simplicity when gains are similar.",
            "If runs regress, back off to a smaller ablation.",
        ],
    }


def _ensure_generated_spec(project_root: Path) -> Path:
    spec_path = _resolve_generated_spec_path(str(project_root))
    if not spec_path.exists():
        print("Scanning project and preparing TensorClaw...", flush=True)
        auto_args = _default_init_args(project_root)
        _write_spec_with_summary(spec_path, auto_args)
        print("TensorClaw is ready.", flush=True)
    return spec_path


def _load_spec_for_project(project_root: Path) -> ResearchSpec:
    spec_path = _ensure_generated_spec(project_root)
    return load_spec(str(spec_path))


def _format_best(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _best_keep_iteration(rows: list[dict[str, str]], direction: str) -> str | None:
    best_value: float | None = None
    best_iteration: str | None = None
    for row in rows:
        if row.get("status") != "keep":
            continue
        value = _safe_float(row.get("metric_value"))
        if value is None:
            continue
        if best_value is None:
            best_value = value
            best_iteration = row.get("iteration")
            continue
        if direction == "maximize" and value > best_value:
            best_value = value
            best_iteration = row.get("iteration")
        if direction != "maximize" and value < best_value:
            best_value = value
            best_iteration = row.get("iteration")
    return best_iteration


def _print_status(spec: ResearchSpec, detailed: bool = True) -> None:
    rows = read_rows(spec.paths.results_tsv)
    best = best_metric(rows, spec.metric.direction)
    print(f"Metric: {spec.metric.name} ({spec.metric.direction}), best: {_format_best(best)}")
    if not rows:
        print("No iterations yet.")
        return
    if not detailed:
        print(f"Iterations: {len(rows)}")
        return

    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    ordered_statuses = [
        "keep",
        "discard",
        "crash",
        "proposal-rejected",
        "proposal-failed",
        "no-change",
        "dry-run",
        "agent-failed",
    ]
    counts_text = " ".join(
        f"{status}={status_counts[status]}" for status in ordered_statuses if status in status_counts
    )
    print(f"Iterations: {len(rows)} {counts_text}".rstrip())

    best_iteration = _best_keep_iteration(rows, spec.metric.direction)
    print("History:")
    for index, row in enumerate(rows):
        iteration = row.get("iteration", "?")
        tags: list[str] = []
        if iteration == best_iteration and row.get("status") == "keep":
            tags.append("best")
        if index == len(rows) - 1:
            tags.append("latest")
        tag_text = f" ({', '.join(tags)})" if tags else ""
        print(
            f"iter={iteration} "
            f"status={row.get('status', 'n/a')} "
            f"metric={row.get('metric_value', 'n/a')} "
            f"commit={row.get('commit', 'n/a')}{tag_text}"
        )


def _tail_text(path: Path, num_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-num_lines:])


def _print_latest_log_tail(spec: ResearchSpec, num_lines: int = 40) -> None:
    rows = read_rows(spec.paths.results_tsv)
    for row in reversed(rows):
        log_path = row.get("log_path", "").strip()
        if not log_path:
            continue
        text = _tail_text(Path(log_path), num_lines)
        print(f"Tail {num_lines}: {log_path}")
        print(text if text else "(log file empty)")
        return
    print("No experiment log found yet.")


def _truncate_for_display(text: str, max_lines: int = 18, max_chars: int = 1400) -> str:
    if not text:
        return ""
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 3].rstrip() + "..."
    return clipped


def _render_chat_history(history: list[tuple[str, str]], limit: int = CHAT_HISTORY_LIMIT) -> str:
    if not history:
        return "(no prior conversation)"
    selected = history[-limit:]
    return "\n".join(f"{role}: {text}" for role, text in selected)


def _parse_chat_decision(raw_text: str) -> ChatDecision:
    payload = _extract_payload_from_json_stream(raw_text) or _extract_json_object(raw_text)
    assistant_reply = str(payload.get("assistant_reply", "")).strip()
    if not assistant_reply:
        assistant_reply = "I can help with the next experiment step."

    action_raw = payload.get("action", {}) if isinstance(payload.get("action"), dict) else {}
    action_type = str(action_raw.get("type", "none")).strip().lower()
    if action_type not in CHAT_ACTIONS:
        action_type = "none"
    direction = str(action_raw.get("direction", "")).strip()

    count_raw = action_raw.get("count", 1)
    tail_raw = action_raw.get("tail_lines", 40)
    try:
        count = int(count_raw)
    except (TypeError, ValueError):
        count = 1
    try:
        tail_lines = int(tail_raw)
    except (TypeError, ValueError):
        tail_lines = 40

    count = max(1, min(count, 20))
    tail_lines = max(1, min(tail_lines, 400))
    return ChatDecision(
        assistant_reply=assistant_reply,
        action_type=action_type,
        direction=direction,
        count=count,
        tail_lines=tail_lines,
    )


def _run_chat_query(spec: ResearchSpec, user_text: str, history: list[tuple[str, str]]) -> ChatDecision:
    command_template = _ensure_json_mode(
        (spec.agent.proposal_command or _derive_proposal_command(spec.agent.command)).strip()
    )
    if not command_template:
        return ChatDecision(assistant_reply="Agent is not configured for chat.", action_type="none")
    if not _has_agent_credentials():
        return ChatDecision(
            assistant_reply="No provider credentials found. Export OPENAI_API_KEY or ANTHROPIC_API_KEY.",
            action_type="none",
        )

    rows = read_rows(spec.paths.results_tsv)
    best = best_metric(rows, spec.metric.direction)
    best_text = _format_best(best)
    recent_results = summarize_recent(rows, limit=5)

    prompt = (
        f"{CHAT_DECISION_PROMPT}\n\n"
        f"Metric: {spec.metric.name} ({spec.metric.direction}), best={best_text}\n"
        f"Recent results:\n{recent_results}\n\n"
        f"Conversation so far:\n{_render_chat_history(history)}\n\n"
        f"Latest user message:\n{user_text}\n"
    )
    rendered = render_template(
        command_template,
        {
            "instruction": prompt,
            "instruction_shell": shell_escape(prompt),
            "instruction_path": "",
        },
    )
    rendered = f"{rendered} </dev/null"
    result = run_command(
        rendered,
        cwd=spec.project_root,
        timeout_seconds=spec.agent.timeout_seconds,
        stream_output=False,
    )
    if result.timed_out:
        return ChatDecision(assistant_reply="Agent chat timed out.", action_type="none")
    if result.returncode != 0:
        excerpt = _tail_text_excerpt(result.stderr or result.stdout)
        reply = f"Agent chat failed (exit {result.returncode})."
        if excerpt:
            reply = f"{reply}\n{excerpt}"
        return ChatDecision(assistant_reply=reply, action_type="none")

    raw_output = f"{result.stdout}\n{result.stderr}"
    try:
        return _parse_chat_decision(raw_output)
    except ValueError:
        text = (result.stdout or "").strip() or (result.stderr or "").strip() or "(No response.)"
        return ChatDecision(assistant_reply=text, action_type="none")


def _reset_history(spec: ResearchSpec) -> None:
    for file_path in [Path(spec.paths.results_tsv), Path(spec.paths.journal_md)]:
        if file_path.exists():
            file_path.unlink()

    for dir_path in [Path(spec.paths.logs_dir), Path(spec.paths.instructions_dir)]:
        if dir_path.exists():
            shutil.rmtree(dir_path)

    ensure_ledger(spec.paths.results_tsv)
    ensure_journal(spec.paths.journal_md, spec.name)
    print("Cleared TensorClaw history for this project.")


def _interactive_proposal_approval(iteration: int, idea: str, proposal_text: str) -> bool:
    print(f"Iteration {iteration} proposal:")
    print(f"- Direction: {idea}")
    excerpt = _truncate_for_display(proposal_text)
    if excerpt:
        print(excerpt)
    else:
        print("(empty proposal)")

    while True:
        try:
            raw = input("Approve this plan? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if raw in {"", "y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y/yes or n/no.")


def _run_single_iteration(
    spec: ResearchSpec,
    dry_run: bool,
    direction: str | None,
    live_output: bool,
    require_proposal_approval: bool,
) -> int:
    old_ideas = list(spec.ideas)
    if direction:
        spec.ideas = [direction]

    try:
        result = run_research_loop(
            spec=spec,
            max_iterations=1,
            dry_run=dry_run,
            live_output=live_output,
            proposal_approval=_interactive_proposal_approval if require_proposal_approval else None,
        )
    finally:
        spec.ideas = old_ideas

    _print_status(spec, detailed=False)
    return result


def _parse_command_iterations(raw: str, default: int) -> int | None:
    pieces = raw.strip().split(maxsplit=1)
    if len(pieces) == 1:
        return default
    try:
        value = int(pieces[1])
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def run_interactive_mode(project_root: Path, dry_run: bool = False) -> int:
    try:
        spec = _load_spec_for_project(project_root)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to initialize TensorClaw: {exc}", file=sys.stderr)
        return 1

    print(f"TensorClaw: {project_root}")
    if dry_run:
        print("Dry-run mode enabled: experiment commands will not execute.")
    print("Type /help for commands. Enter runs one iteration; text chats with the agent.")
    _print_status(spec, detailed=False)
    chat_history: list[tuple[str, str]] = []

    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if raw in {"/quit", "/exit"}:
            return 0

        if raw in {"/help", "?"}:
            print(INTERACTIVE_HELP.rstrip())
            continue

        if raw == "/status":
            _print_status(spec)
            continue

        if raw == "/reset":
            try:
                confirm = input("Clear TensorClaw history here? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if confirm in {"y", "yes"}:
                _reset_history(spec)
                _print_status(spec, detailed=False)
            else:
                print("Reset canceled.")
            continue

        if raw.startswith("/tail"):
            pieces = raw.split(maxsplit=1)
            num_lines = 40
            if len(pieces) == 2:
                try:
                    num_lines = int(pieces[1])
                except ValueError:
                    print("/tail expects an integer line count.")
                    continue
                if num_lines <= 0:
                    print("/tail line count must be positive.")
                    continue
            _print_latest_log_tail(spec, num_lines=num_lines)
            continue

        if raw.startswith("/run"):
            iterations = _parse_command_iterations(raw, default=1)
            if iterations is None:
                print("Expected a positive integer iteration count.")
                continue
            for _ in range(iterations):
                rc = _run_single_iteration(
                    spec=spec,
                    dry_run=dry_run,
                    direction=None,
                    live_output=True,
                    require_proposal_approval=True,
                )
                if rc != 0:
                    print("Iteration failed. Interactive mode is still active.")
                    break
            continue

        if raw == "":
            rc = _run_single_iteration(
                spec=spec,
                dry_run=dry_run,
                direction=None,
                live_output=True,
                require_proposal_approval=True,
            )
            if rc != 0:
                print("Iteration failed. Interactive mode is still active.")
            continue

        decision = _run_chat_query(spec, raw, chat_history)
        print(decision.assistant_reply)
        chat_history.append(("user", raw))
        chat_history.append(("assistant", decision.assistant_reply))
        if len(chat_history) > CHAT_HISTORY_LIMIT * 2:
            chat_history = chat_history[-CHAT_HISTORY_LIMIT * 2 :]

        if decision.action_type == "run_iteration":
            rc = _run_single_iteration(
                spec=spec,
                dry_run=dry_run,
                direction=decision.direction or None,
                live_output=True,
                require_proposal_approval=True,
            )
            if rc != 0:
                print("Iteration failed. Interactive mode is still active.")
        elif decision.action_type == "run_iterations":
            for _ in range(decision.count):
                rc = _run_single_iteration(
                    spec=spec,
                    dry_run=dry_run,
                    direction=decision.direction or None,
                    live_output=True,
                    require_proposal_approval=True,
                )
                if rc != 0:
                    print("Iteration failed. Interactive mode is still active.")
                    break
        elif decision.action_type == "status":
            _print_status(spec)
        elif decision.action_type == "tail":
            _print_latest_log_tail(spec, num_lines=decision.tail_lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorclaw",
        description="Interactive pi-powered research harness.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to research project root (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run interactive loop without executing experiment command",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser(
        "run",
        help="Run autonomous loop (headless mode)",
    )
    run_parser.add_argument(
        "--project-root",
        default=".",
        help="Path to research project root (default: current directory)",
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations for this run",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run setup/agent/loop logic without executing experiment command",
    )
    run_parser.add_argument(
        "--no-live-output",
        action="store_true",
        help="Disable live streaming of agent/experiment output",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        project_root = Path(args.project_root).expanduser().resolve()
        return run_interactive_mode(project_root=project_root, dry_run=bool(args.dry_run))

    if args.command == "run":
        project_root = Path(args.project_root).expanduser().resolve()
        print(f"TensorClaw run: {project_root}")

        try:
            spec = _load_spec_for_project(project_root)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load TensorClaw config: {exc}", file=sys.stderr)
            return 1

        return run_research_loop(
            spec=spec,
            max_iterations=args.max_iterations,
            dry_run=args.dry_run,
            live_output=not args.no_live_output,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
