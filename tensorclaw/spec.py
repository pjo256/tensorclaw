from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional



DEFAULT_INSTRUCTION_TEMPLATE = """You are running autonomous experiment iteration {iteration}.

Objective:
- Optimize {metric_name} ({metric_direction})
- Current best: {best_metric}

Scope:
- Only edit files listed in target_files.
- Keep changes minimal and hypothesis-driven.

Target files:
{target_files_text}

Recent experiment summary:
{recent_results}

Recent crash diagnostics:
{recent_crash_diagnostics}

Current hypothesis:
{idea}

Approved plan (from proposal phase):
{approved_plan}

Output requirements:
1) Apply one coherent experimental change.
2) If recent crashes are shown, first restore a runnable experiment that reaches metric output.
3) Keep code runnable with existing dependencies.
4) Do not modify generated artifacts/logs.
"""

DEFAULT_PROPOSAL_TEMPLATE = """You are preparing experiment iteration {iteration}.

Objective:
- Optimize {metric_name} ({metric_direction})
- Current best: {best_metric}

Scope:
- Read project files and recent results.
- Do not edit files in proposal mode.

Target files:
{target_files_text}

Recent experiment summary:
{recent_results}

Recent crash diagnostics:
{recent_crash_diagnostics}

Current hypothesis:
{idea}

Output format (plain text):
- Keep it short (max 8 lines).
- Start with: Plan: <one sentence>.
- Then: Edits:
  - <file>: <change>
- End with:
  - Impact: <one sentence>
  - Risk: <one sentence>
"""


@dataclass
class MetricSpec:
    name: str = "val_bpb"
    direction: str = "minimize"
    pattern: str = r"^val_bpb:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    baseline: Optional[float] = None
    min_delta: float = 0.0


@dataclass
class ConstraintSpec:
    name: str
    pattern: str
    op: str
    threshold: float


@dataclass
class MemorySpec:
    pattern: Optional[str] = None
    scale_to_gb: float = 1.0 / 1024.0


@dataclass
class CommandSpec:
    experiment: str
    setup: list[str] = field(default_factory=list)


@dataclass
class AgentSpec:
    enabled: bool = False
    command: str = ""
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE
    proposal_enabled: bool = True
    proposal_command: str = ""
    proposal_instruction_template: str = DEFAULT_PROPOSAL_TEMPLATE
    timeout_seconds: int = 300
    save_instruction: bool = True
    continue_on_failure: bool = True


@dataclass
class GitSpec:
    enabled: bool = True
    auto_commit: bool = True
    commit_message_template: str = "tensorclaw iter {iteration}: {idea}"
    discard_strategy: str = "hard-reset"
    revert_on_crash: bool = True
    exclude_paths: list[str] = field(default_factory=list)


@dataclass
class PathsSpec:
    results_tsv: str
    journal_md: str
    logs_dir: str
    instructions_dir: str


@dataclass
class LoopSpec:
    max_iterations: int = 20
    timeout_seconds: int = 600


@dataclass
class ResearchSpec:
    version: int
    name: str
    project_root: str
    target_files: list[str]
    commands: CommandSpec
    metric: MetricSpec
    constraints: list[ConstraintSpec]
    memory: Optional[MemorySpec]
    agent: AgentSpec
    git: GitSpec
    paths: PathsSpec
    loop: LoopSpec
    ideas: list[str]
    spec_path: str = ""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    return [str(value)]


def _resolve(path_or_none: Optional[str], base: Path) -> Optional[str]:
    if path_or_none is None:
        return None
    path = Path(path_or_none).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def _resolve_many(paths: list[str], base: Path) -> list[str]:
    resolved = []
    for item in paths:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        resolved.append(str(path))
    return resolved


def _load_raw(path: str) -> dict[str, Any]:
    spec_path = Path(path).expanduser().resolve()
    with spec_path.open("r", encoding="utf-8") as f:
        if spec_path.suffix.lower() in {".json"}:
            raw = json.load(f)
        else:
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "PyYAML is required for YAML specs. Install dependencies first (pip install -e .)."
                ) from exc
            raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Spec root must be a mapping/object")
    return raw


def load_spec(path: str) -> ResearchSpec:
    raw = _load_raw(path)
    spec_path = Path(path).expanduser().resolve()
    spec_dir = spec_path.parent

    version = int(raw.get("version", 1))
    name = str(raw.get("name", "tensorclaw-run"))

    project_root_raw = raw.get("project_root", str(spec_dir))
    project_root = Path(_resolve(str(project_root_raw), spec_dir) or str(spec_dir)).resolve()

    commands_raw = raw.get("commands", {}) or {}
    experiment_command = commands_raw.get("experiment") or raw.get("experiment_command")
    if not experiment_command:
        raise ValueError("Spec must include commands.experiment")
    setup_commands = _as_list(commands_raw.get("setup", []))

    metric_raw = raw.get("metric", {}) or {}
    metric_direction = str(metric_raw.get("direction", "minimize")).strip().lower()
    if metric_direction not in {"minimize", "maximize"}:
        raise ValueError("metric.direction must be 'minimize' or 'maximize'")

    metric_baseline = metric_raw.get("baseline")
    metric = MetricSpec(
        name=str(metric_raw.get("name", "val_bpb")),
        direction=metric_direction,
        pattern=str(metric_raw.get("pattern", MetricSpec.pattern)),
        baseline=float(metric_baseline) if metric_baseline is not None else None,
        min_delta=float(metric_raw.get("min_delta", 0.0)),
    )

    constraints: list[ConstraintSpec] = []
    for item in raw.get("constraints", []) or []:
        constraints.append(
            ConstraintSpec(
                name=str(item["name"]),
                pattern=str(item["pattern"]),
                op=str(item.get("op", "<=")),
                threshold=float(item["threshold"]),
            )
        )

    memory_raw = raw.get("memory")
    memory = None
    if isinstance(memory_raw, dict) and memory_raw.get("pattern"):
        memory = MemorySpec(
            pattern=str(memory_raw.get("pattern")),
            scale_to_gb=float(memory_raw.get("scale_to_gb", 1.0 / 1024.0)),
        )

    target_files = _as_list(raw.get("target_files") or raw.get("targets") or [])

    default_paths = {
        "results_tsv": ".tensorclaw/results.tsv",
        "journal_md": ".tensorclaw/journal.md",
        "logs_dir": ".tensorclaw/logs",
        "instructions_dir": ".tensorclaw/instructions",
    }
    paths_raw = raw.get("paths", {}) or {}
    paths = PathsSpec(
        results_tsv=_resolve(str(paths_raw.get("results_tsv", default_paths["results_tsv"])), project_root) or "",
        journal_md=_resolve(str(paths_raw.get("journal_md", default_paths["journal_md"])), project_root) or "",
        logs_dir=_resolve(str(paths_raw.get("logs_dir", default_paths["logs_dir"])), project_root) or "",
        instructions_dir=_resolve(
            str(paths_raw.get("instructions_dir", default_paths["instructions_dir"])),
            project_root,
        )
        or "",
    )

    loop_raw = raw.get("loop", {}) or {}
    loop = LoopSpec(
        max_iterations=int(loop_raw.get("max_iterations", raw.get("max_iterations", 20))),
        timeout_seconds=int(loop_raw.get("timeout_seconds", raw.get("timeout_seconds", 600))),
    )

    agent_raw = raw.get("agent", {}) or {}
    agent = AgentSpec(
        enabled=bool(agent_raw.get("enabled", False)),
        command=str(agent_raw.get("command", "")),
        instruction_template=str(agent_raw.get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE)),
        proposal_enabled=bool(agent_raw.get("proposal_enabled", True)),
        proposal_command=str(agent_raw.get("proposal_command", "")),
        proposal_instruction_template=str(
            agent_raw.get("proposal_instruction_template", DEFAULT_PROPOSAL_TEMPLATE)
        ),
        timeout_seconds=int(agent_raw.get("timeout_seconds", 300)),
        save_instruction=bool(agent_raw.get("save_instruction", True)),
        continue_on_failure=bool(agent_raw.get("continue_on_failure", True)),
    )
    if agent.enabled and not agent.command:
        raise ValueError("agent.enabled=true requires agent.command")

    git_raw = raw.get("git", {}) or {}
    discard_strategy = str(git_raw.get("discard_strategy", "hard-reset"))
    if discard_strategy not in {"hard-reset", "none"}:
        raise ValueError("git.discard_strategy must be one of: hard-reset, none")
    explicit_excludes = _as_list(git_raw.get("exclude_paths", []))
    git = GitSpec(
        enabled=bool(git_raw.get("enabled", True)),
        auto_commit=bool(git_raw.get("auto_commit", True)),
        commit_message_template=str(git_raw.get("commit_message_template", "tensorclaw iter {iteration}: {idea}")),
        discard_strategy=discard_strategy,
        revert_on_crash=bool(git_raw.get("revert_on_crash", True)),
        exclude_paths=_resolve_many(explicit_excludes, project_root),
    )

    ideas = _as_list(raw.get("ideas", []))
    if not ideas:
        ideas = ["Try one focused experiment that could improve the objective metric."]

    return ResearchSpec(
        version=version,
        name=name,
        project_root=str(project_root),
        target_files=target_files,
        commands=CommandSpec(experiment=str(experiment_command), setup=setup_commands),
        metric=metric,
        constraints=constraints,
        memory=memory,
        agent=agent,
        git=git,
        paths=paths,
        loop=loop,
        ideas=ideas,
        spec_path=str(spec_path),
    )
