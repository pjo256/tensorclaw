from __future__ import annotations

from collections import deque

from rich.markup import escape
from textual.widgets import RichLog


class OutputView(RichLog):
    def __init__(self, max_lines: int = 500) -> None:
        super().__init__(id="output-pane", highlight=False, markup=True, wrap=True, auto_scroll=True)
        self._max_lines = max_lines
        self._buffer: deque[str] = deque(maxlen=max_lines)

    def append_output(self, phase: str, stream: str, line: str) -> None:
        clean = escape(line.rstrip("\n"))
        if not clean:
            return
        prefix = f"[{phase}]"
        if stream == "stderr":
            rendered = f"[red]{prefix} {clean}[/red]"
        elif stream == "meta":
            rendered = f"[yellow]{prefix} {clean}[/yellow]"
        else:
            rendered = f"[cyan]{prefix}[/cyan] {clean}"

        self._buffer.append(rendered)
        if len(self._buffer) == self._max_lines:
            self.clear()
            for item in self._buffer:
                self.write(item)
            return
        self.write(rendered)
