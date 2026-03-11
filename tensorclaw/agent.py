from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .shell import run_command
from .templates import render_template, shell_escape


@dataclass
class AgentRunResult:
    attempted: bool
    success: bool
    command: str
    returncode: int
    timed_out: bool
    instruction: str
    instruction_path: str
    log_path: str
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def run_agent_iteration(
    project_root: str,
    agent_enabled: bool,
    agent_command_template: str,
    instruction_template: str,
    instruction_dir: str,
    logs_dir: str,
    timeout_seconds: int,
    iteration: int,
    context: dict[str, Any],
    phase: str = "apply",
    save_instruction: bool = True,
    stream_output: bool = False,
    raw_stream_output: bool = True,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
) -> AgentRunResult:
    if not agent_enabled:
        return AgentRunResult(
            attempted=False,
            success=True,
            command="",
            returncode=0,
            timed_out=False,
            instruction="",
            instruction_path="",
            log_path="",
            stdout="",
            stderr="",
        )

    instruction = render_template(instruction_template, context)

    instruction_path = ""
    if save_instruction:
        Path(instruction_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(instruction_dir) / f"instruction_{iteration:04d}_{phase}.md"
        file_path.write_text(instruction, encoding="utf-8")
        instruction_path = str(file_path)

    command_context = dict(context)
    command_context["instruction"] = instruction
    command_context["instruction_shell"] = shell_escape(instruction)
    command_context["instruction_path"] = instruction_path

    rendered_command = render_template(agent_command_template, command_context)

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = str(Path(logs_dir) / f"agent_{iteration:04d}_{phase}.log")
    result = run_command(
        rendered_command,
        cwd=project_root,
        timeout_seconds=timeout_seconds,
        log_path=log_path,
        stream_output=stream_output,
        raw_stream_output=raw_stream_output,
        on_stdout_line=on_stdout_line,
        on_stderr_line=on_stderr_line,
    )

    success = (result.returncode == 0) and (not result.timed_out)
    error = ""
    if not success:
        if result.timed_out:
            error = "agent command timed out"
        else:
            error = "agent command failed"

    return AgentRunResult(
        attempted=True,
        success=success,
        command=rendered_command,
        returncode=result.returncode,
        timed_out=result.timed_out,
        instruction=instruction,
        instruction_path=instruction_path,
        log_path=log_path,
        stdout=result.stdout,
        stderr=result.stderr,
        error=error,
    )
