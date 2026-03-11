from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from ..metrics import extract_first_float
from ..shell import run_command
from ..spec import ResearchSpec, load_spec
from ..templates import render_template, shell_escape
from .events import DiscoveryEvent, OutputLineEvent, StartupEvent


Emitter = Callable[[object], None] | None


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
AGENT_TOOLS_FLAG_RE = re.compile(r"(--tools\s+)([^\s]+)")
AGENT_MODE_FLAG_RE = re.compile(r"(--mode\s+)([^\s]+)")
DISCOVERY_TIMEOUT_SECONDS = 120
COMMAND_PROBE_TIMEOUT_SECONDS = 20
DISCOVERY_ATTEMPTS = 2
READ_ONLY_TOOLS = "read,grep,find,ls"

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

DISCOVERY_PROMPT = """You are configuring TensorClaw for this repository.

Read the codebase and return ONLY a JSON object (no prose) with:
- target_files: list of key editable files for training loop changes (paths relative to repo root)
- experiment_command: one command string runnable from repo root that launches the main train/eval run
- metric_name: scalar metric to optimize (e.g. val_bpb, val_loss, accuracy)
- metric_direction: either \"minimize\" or \"maximize\"
- metric_pattern: regex with one capture group for the metric value in run logs

Constraints:
- Use a command that matches the normal experiment flow in this repo.
- Prefer objective validation metrics over training loss.
- Paths must be repo-relative.
- Return valid JSON only.
"""


def _emit(emit: Emitter, event: object) -> None:
    if emit:
        emit(event)


def _load_yaml_module():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required. Install dependencies first (pip install -e .).") from exc
    return yaml


def _resolve_generated_spec_path(project_root: Path) -> Path:
    return project_root / GENERATED_SPEC_REL_PATH


def _has_agent_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _discover_agent_command(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    if has_anthropic and not has_openai:
        return DEFAULT_ANTHROPIC_AGENT_COMMAND
    return DEFAULT_OPENAI_AGENT_COMMAND


def _derive_proposal_command(agent_command: str) -> str:
    if AGENT_TOOLS_FLAG_RE.search(agent_command):
        return AGENT_TOOLS_FLAG_RE.sub(lambda m: f"{m.group(1)}{READ_ONLY_TOOLS}", agent_command, count=1)
    return f"{agent_command} --tools {READ_ONLY_TOOLS}"


def _ensure_json_mode(agent_command: str) -> str:
    if AGENT_MODE_FLAG_RE.search(agent_command):
        return AGENT_MODE_FLAG_RE.sub(lambda m: f"{m.group(1)}json", agent_command, count=1)
    return f"{agent_command} --mode json"


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


def _tail_text_excerpt(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


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


def _normalize_experiment_command(command: str) -> str:
    normalized = command.strip()
    if not normalized:
        return normalized
    py = shell_escape(sys.executable)
    return re.sub(r"^\s*python3?\b", py, normalized, count=1)


def _validate_experiment_command(project_root: Path, command: str) -> None:
    probe = run_command(command, cwd=str(project_root), timeout_seconds=COMMAND_PROBE_TIMEOUT_SECONDS, stream_output=False)
    if probe.timed_out or probe.returncode == 0:
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


def _read_text_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _infer_baseline_from_log(project_root: Path, metric_pattern: str) -> float | None:
    baseline_log = _read_text_if_exists(project_root / "run_baseline.log")
    if not baseline_log:
        return None
    return extract_first_float(baseline_log, metric_pattern)


def _run_agent_discovery(project_root: Path, agent_command: str, feedback: str = "", emit: Emitter = None) -> dict[str, Any]:
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
    rendered = f"{rendered} </dev/null"

    log_path = project_root / ".tensorclaw" / "logs" / "agent_discovery.log"

    def _stdout(line: str) -> None:
        _emit(emit, OutputLineEvent(phase="discovery", stream="stdout", line=line.rstrip("\n")))

    def _stderr(line: str) -> None:
        _emit(emit, OutputLineEvent(phase="discovery", stream="stderr", line=line.rstrip("\n")))

    result = run_command(
        rendered,
        cwd=str(project_root),
        timeout_seconds=DISCOVERY_TIMEOUT_SECONDS,
        log_path=str(log_path),
        stream_output=True,
        raw_stream_output=False,
        on_stdout_line=_stdout,
        on_stderr_line=_stderr,
    )
    if result.timed_out:
        raise ValueError("Agent discovery timed out.")
    if result.returncode != 0:
        excerpt = _tail_text_excerpt(result.stderr or result.stdout)
        raise ValueError(f"Agent discovery failed (exit {result.returncode}).\n{excerpt}")

    raw_output = f"{result.stdout}\n{result.stderr}"
    payload = _extract_payload_from_json_stream(raw_output) or _extract_json_object(raw_output)
    return payload


def build_generated_spec(project_root: Path, emit: Emitter = None) -> dict[str, Any]:
    agent_command = _discover_agent_command()
    target_files: list[str] = []
    experiment_command = ""
    metric_name = "val_bpb"
    metric_direction = "minimize"
    metric_pattern = DEFAULT_METRIC_PATTERN
    last_error = ""

    for attempt in range(1, DISCOVERY_ATTEMPTS + 1):
        _emit(emit, DiscoveryEvent(f"discovery attempt {attempt}/{DISCOVERY_ATTEMPTS}"))
        discovery = _run_agent_discovery(project_root, agent_command, feedback=last_error, emit=emit)

        target_files = _validated_target_files(project_root, discovery.get("target_files"))
        if not target_files:
            last_error = "target_files were missing or invalid."
            continue
        _emit(emit, DiscoveryEvent(f"candidate targets={','.join(target_files)}"))

        experiment_command = _normalize_experiment_command(str(discovery.get("experiment_command", "")).strip())
        if not experiment_command:
            last_error = "experiment_command was missing."
            continue
        _emit(emit, DiscoveryEvent(f"candidate command={experiment_command}"))

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
        _emit(emit, DiscoveryEvent(f"candidate metric={metric_name} ({metric_direction})"))

        try:
            _validate_experiment_command(project_root, experiment_command)
        except ValueError as exc:
            last_error = str(exc)
            continue
        _emit(emit, DiscoveryEvent("discovery validation passed"))
        break
    else:
        detail = f"\nlast_error: {last_error}" if last_error else ""
        raise ValueError(f"Agent discovery failed validation after {DISCOVERY_ATTEMPTS} attempts.{detail}")

    baseline = _infer_baseline_from_results_tsv(project_root, metric_name, metric_direction)
    if baseline is None:
        baseline = _infer_baseline_from_log(project_root, metric_pattern)

    metric: dict[str, Any] = {
        "name": metric_name,
        "direction": metric_direction,
        "pattern": metric_pattern,
        "min_delta": 0.0001,
    }
    if baseline is not None:
        metric["baseline"] = baseline

    proposal_command = _derive_proposal_command(agent_command)

    return {
        "version": 1,
        "name": project_root.name,
        "project_root": str(project_root),
        "target_files": target_files,
        "commands": {"experiment": experiment_command},
        "metric": metric,
        "memory": {
            "pattern": DEFAULT_MEMORY_PATTERN,
            "scale_to_gb": 0.0009765625,
        },
        "loop": {
            "max_iterations": 20,
            "timeout_seconds": 1200,
        },
        "agent": {
            "enabled": _has_agent_credentials(),
            "command": agent_command,
            "proposal_enabled": True,
            "proposal_command": proposal_command,
            "timeout_seconds": 600,
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


def write_generated_spec(path: Path, raw: dict[str, Any]) -> None:
    yaml = _load_yaml_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def ensure_generated_spec(project_root: Path, emit: Emitter = None) -> Path:
    spec_path = _resolve_generated_spec_path(project_root)
    if spec_path.exists():
        return spec_path
    _emit(emit, StartupEvent(stage="init", message="scanning project and preparing TensorClaw"))
    raw = build_generated_spec(project_root, emit=emit)
    write_generated_spec(spec_path, raw)
    _emit(emit, StartupEvent(stage="init", message=f"generated config: {spec_path}"))
    return spec_path


def load_spec_for_project(project_root: Path, emit: Emitter = None) -> ResearchSpec:
    spec_path = ensure_generated_spec(project_root, emit=emit)
    return load_spec(str(spec_path))
