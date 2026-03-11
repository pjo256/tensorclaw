from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    log_path: Optional[str] = None


def _write_log(log_path: str, command: str, result: CommandResult) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"$ {command}\n")
        f.write(f"returncode: {result.returncode}\n")
        f.write(f"timed_out: {result.timed_out}\n")
        f.write(f"duration_seconds: {result.duration_seconds:.3f}\n")
        f.write("\n[stdout]\n")
        f.write(result.stdout)
        f.write("\n\n[stderr]\n")
        f.write(result.stderr)
        f.write("\n")


def run_command(
    command: str,
    cwd: str,
    timeout_seconds: Optional[int] = None,
    log_path: Optional[str] = None,
    stream_output: bool = False,
    raw_stream_output: bool = True,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
) -> CommandResult:
    start = time.time()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if not stream_output:
        timed_out = False
        stdout = ""
        stderr = ""
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
    else:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        def _drain_pipe(
            pipe,
            parts: list[str],
            sink,
            callback: Callable[[str], None] | None,
        ) -> None:
            if pipe is None:
                return
            for line in iter(pipe.readline, ""):
                if line == "":
                    break
                parts.append(line)
                if raw_stream_output:
                    sink.write(line)
                    sink.flush()
                if callback is not None:
                    callback(line.rstrip("\r\n"))
            pipe.close()

        stdout_thread = threading.Thread(
            target=_drain_pipe,
            args=(process.stdout, stdout_parts, sys.stdout, on_stdout_line),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_pipe,
            args=(process.stderr, stderr_parts, sys.stderr, on_stderr_line),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()

        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

    duration_seconds = time.time() - start
    result = CommandResult(
        command=command,
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=duration_seconds,
        log_path=log_path,
    )

    if log_path:
        _write_log(log_path, command, result)

    return result


def run_git(args: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return completed
