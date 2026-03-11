from __future__ import annotations

from rich.markup import escape
from textual.widgets import Static

from ...engine.models import IterationRecord


class IterationsView(Static):
    def __init__(self) -> None:
        super().__init__(id="iterations-pane")

    def set_iterations(self, records: list[IterationRecord]) -> None:
        if not records:
            self.update("[b]ITERATIONS[/b]\n(no runs yet)")
            return

        lines = ["[b]ITERATIONS[/b]"]
        header = f"{'iter':>4}  {'status':<12}  {'metric':>10}  {'commit':<8}"
        lines.append(f"[#d6b35f]{escape(header)}[/#d6b35f]")

        for record in records:
            metric = "n/a" if record.metric_value is None else f"{record.metric_value:.6f}"
            status = record.status
            if record.iteration == 0 and record.description.strip().lower() == "baseline":
                status = "baseline"
            if record.is_best:
                status = f"{status} ★"
            if record.is_latest:
                status = f"{status} •"
            row = f"{record.iteration:>4}  {status:<12}  {metric:>10}  {(record.commit or '')[:8]:<8}"
            lines.append(f"[{self._status_color(record.status)}]{escape(row)}[/{self._status_color(record.status)}]")

        self.update("\n".join(lines))

    @staticmethod
    def _status_color(status: str) -> str:
        normalized = status.strip().lower()
        if normalized == "keep":
            return "#79d38a"
        if normalized == "discard":
            return "#f2b06f"
        if normalized == "crash":
            return "#ef6d64"
        if normalized in {"baseline", "dry-run", "dry_run"}:
            return "#6ec6d9"
        return "#c7ced8"
