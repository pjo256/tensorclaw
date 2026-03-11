from __future__ import annotations

from textual.widgets import Static


class Sparkline:
    BLOCKS = "▁▂▃▄▅▆▇█"

    @classmethod
    def render(cls, values: list[float], width: int | None = None) -> str:
        if not values:
            return ""
        if width is None:
            width = len(values)
        if width <= 0:
            return ""

        if len(values) == 1:
            return cls.BLOCKS[len(cls.BLOCKS) // 2] * width

        # Always render to full panel width so short histories remain legible.
        sample: list[float]
        if len(values) == width:
            sample = values
        else:
            sample = []
            last = len(values) - 1
            for i in range(width):
                position = (i / float(width - 1)) * last
                left = int(position)
                right = min(left + 1, last)
                frac = position - left
                value = values[left] * (1.0 - frac) + values[right] * frac
                sample.append(value)

        low = min(sample)
        high = max(sample)
        span = high - low
        if span == 0:
            return cls.BLOCKS[len(cls.BLOCKS) // 2] * len(sample)

        chars: list[str] = []
        for value in sample:
            idx = int(round((value - low) / span * (len(cls.BLOCKS) - 1)))
            idx = max(0, min(len(cls.BLOCKS) - 1, idx))
            chars.append(cls.BLOCKS[idx])
        return "".join(chars)


class MetricsView(Static):
    def __init__(self) -> None:
        super().__init__(id="metrics-pane")
        self.metric_name = ""
        self.metric_direction = "minimize"
        self.baseline: float | None = None
        self.best: float | None = None
        self.current: float | None = None
        self.series: list[float] = []
        self.aux_series: dict[str, list[float]] = {}
        self.aux_latest: dict[str, float] = {}
        self.running_iteration: int | None = None
        self.running_commit: str = ""
        self.running_phase: str = "idle"
        self.token_input_total: int = 0
        self.token_output_total: int = 0
        self.token_total: int = 0
        self.last_token_total: int | None = None
        self.last_token_source: str = ""

    def set_metric_info(
        self,
        *,
        metric_name: str,
        direction: str,
        baseline: float | None,
        best: float | None,
        current: float | None = None,
    ) -> None:
        self.metric_name = metric_name
        self.metric_direction = direction
        self.baseline = baseline
        self.best = best
        self.current = current if current is not None else self.current
        self.refresh_view()

    def set_series(self, values: list[float]) -> None:
        self.series = values[-120:]
        self.current = self.series[-1] if self.series else None
        self.refresh_view()

    def set_aux_series(self, series: dict[str, list[float]]) -> None:
        self.aux_series = {name: values[-120:] for name, values in series.items() if values}
        self.aux_latest = {name: values[-1] for name, values in self.aux_series.items() if values}
        metric_key = _normalize_metric_name(self.metric_name)
        if self.metric_name in self.aux_latest:
            self.current = self.aux_latest[self.metric_name]
        elif metric_key and metric_key in self.aux_latest:
            self.current = self.aux_latest[metric_key]
        self.refresh_view()

    def set_run_context(self, *, iteration: int | None, commit: str, phase: str) -> None:
        self.running_iteration = iteration
        self.running_commit = commit
        self.running_phase = phase
        self.refresh_view()

    def set_token_usage(
        self,
        *,
        input_total: int,
        output_total: int,
        total: int,
        last_total: int | None,
        source: str,
    ) -> None:
        self.token_input_total = max(0, int(input_total))
        self.token_output_total = max(0, int(output_total))
        self.token_total = max(0, int(total))
        self.last_token_total = last_total if last_total is None else max(0, int(last_total))
        self.last_token_source = source
        self.refresh_view()

    def record_metrics(self, metrics: dict[str, float]) -> None:
        metric_key = _normalize_metric_name(self.metric_name)
        for name, value in metrics.items():
            points = self.aux_series.setdefault(name, [])
            points.append(value)
            if len(points) > 120:
                del points[:-120]
            self.aux_latest[name] = value
            if name == self.metric_name or (metric_key and name == metric_key):
                self.current = value
                self.series.append(value)
                if len(self.series) > 120:
                    del self.series[:-120]
        self.refresh_view()

    def push_value(self, value: float | None) -> None:
        if value is not None:
            self.current = value
            self.series.append(value)
            self.series = self.series[-120:]
        self.refresh_view()

    def update_best(self, best: float | None) -> None:
        self.best = best
        self.refresh_view()

    def refresh_view(self) -> None:
        if not self.metric_name:
            self.update("[b]Metrics[/b]\nNo metric configured.")
            return

        chart = Sparkline.render(self.series, width=30) if self.series else "(no points yet)"
        current_text = "n/a" if self.current is None else f"{self.current:.6f}"
        best_text = "n/a" if self.best is None else f"{self.best:.6f}"
        baseline_text = "n/a" if self.baseline is None else f"{self.baseline:.6f}"
        trend = self._trend_label()
        lines = [
            "[b]METRICS[/b]",
            f"{self.metric_name} ({self.metric_direction})",
            chart,
            f"current: {current_text}",
            f"best: {best_text}",
            f"baseline: {baseline_text}",
            f"trend: {trend}",
        ]

        run_lines = self._run_lines()
        if run_lines:
            lines.append("")
            lines.append("[b]run[/b]")
            lines.extend(run_lines)

        live_lines = self._live_metric_lines(max_rows=7)
        if live_lines:
            lines.append("")
            lines.append("[b]live monitor[/b]")
            lines.extend(live_lines)

        token_lines = self._token_lines()
        if token_lines:
            lines.append("")
            lines.append("[b]pi usage[/b]")
            lines.extend(token_lines)

        self.update("\n".join(lines))

    def _trend_label(self) -> str:
        if len(self.series) < 2:
            return "neutral"
        prev = self.series[-2]
        cur = self.series[-1]
        delta = cur - prev
        eps = 1e-12
        if abs(delta) <= eps:
            return "neutral"
        improving = delta < 0 if self.metric_direction == "minimize" else delta > 0
        if improving:
            return "[green]improving[/green]"
        return "[red]regressing[/red]"

    def _live_metric_lines(self, max_rows: int) -> list[str]:
        if not self.aux_latest:
            return []

        preferred = [
            "step",
            "train_loss",
            "val_loss",
            "val_bpb",
            "tok_per_sec",
            "grad_norm",
            "peak_vram_mb",
            "num_steps",
            "num_params_m",
            "training_seconds",
            "total_seconds",
        ]
        names = sorted(self.aux_latest.keys(), key=lambda name: (preferred.index(name) if name in preferred else len(preferred), name))
        lines: list[str] = []
        for name in names:
            if name == self.metric_name:
                continue
            series = self.aux_series.get(name, [])
            if not series:
                continue
            spark = Sparkline.render(series, width=12)
            latest = self.aux_latest[name]
            lines.append(f"{name:<12} {spark} {self._fmt_value(latest)}")
            if len(lines) >= max_rows:
                break
        return lines

    def _run_lines(self) -> list[str]:
        if self.running_iteration is None and not self.running_commit and self.running_phase in {"", "idle"}:
            return []
        lines: list[str] = []
        if self.running_iteration is not None:
            lines.append(f"iter: {self.running_iteration}")
        if self.running_commit:
            lines.append(f"commit: {self.running_commit[:8]}")
        if self.running_phase:
            lines.append(f"phase: {self.running_phase}")
        return lines

    def _token_lines(self) -> list[str]:
        lines = [
            f"total: {self.token_total}",
            f"in/out: {self.token_input_total}/{self.token_output_total}",
        ]
        if self.last_token_total is not None:
            source = self.last_token_source or "agent"
            lines.append(f"last {source}: {self.last_token_total}")
        else:
            lines.append("last: n/a")
        return lines

    @staticmethod
    def _fmt_value(value: float) -> str:
        if abs(value) >= 10000:
            return f"{value:,.0f}"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.4g}"


def _normalize_metric_name(name: str) -> str:
    if not name:
        return ""
    normalized = name.strip().lower()
    normalized = normalized.replace("/", "_per_")
    normalized = normalized.replace("%", "_pct")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    return "".join(ch for ch in normalized if ch.isalnum() or ch == "_").strip("_")
