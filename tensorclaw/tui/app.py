from __future__ import annotations

import queue
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Static

from ..engine.controller import ResearchController
from ..engine.events import (
    ChatEvent,
    ChatStreamEvent,
    DecisionEvent,
    DiscoveryEvent,
    EngineEvent,
    ErrorEvent,
    IterationEvent,
    MetricEvent,
    OutputLineEvent,
    ProposalEvent,
    StartupEvent,
    TelemetryEvent,
    UsageEvent,
)
from .widgets.chat_view import ChatView
from .widgets.iterations_view import IterationsView
from .widgets.metrics_view import MetricsView
from .widgets.output_view import OutputView
from .widgets.status_bar import StatusBar


class TensorClawApp(App[None]):
    CSS_PATH = "theme.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("y", "approve_plan", "Approve Plan"),
        Binding("n", "reject_plan", "Reject Plan"),
    ]

    def __init__(self, project_root: Path, dry_run: bool = False) -> None:
        super().__init__()
        self._project_root = project_root
        self._event_queue: queue.SimpleQueue[EngineEvent] = queue.SimpleQueue()
        self._controller = ResearchController(project_root=project_root, dry_run=dry_run, event_sink=self._push_event)
        self._init_busy = False
        self._chat_busy = False
        self._iteration_busy = False
        self._rendered_history = False

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(id="top-row"):
            yield ChatView()
            with Vertical(id="right-pane"):
                yield MetricsView()
                yield IterationsView()
        yield OutputView()
        yield Input(placeholder="Talk to TensorClaw. Ask questions or request an experiment proposal.", id="input")
        yield StatusBar()

    def on_mount(self) -> None:
        self.query_one(OutputView).append_output("startup", "meta", f"TensorClaw: {self._project_root}")
        self.set_interval(0.02, self._drain_events)
        self.call_after_refresh(self._focus_input)
        self._start_background("initialize", self._controller.initialize)

    def _focus_input(self) -> None:
        self.set_focus(self.query_one(Input))

    def action_approve_plan(self) -> None:
        if self._init_busy or self._iteration_busy:
            return
        if self._chat_busy:
            self.query_one(OutputView).append_output("ui", "meta", "Wait for the current chat response to finish.")
            return
        if not self._controller.state.has_pending_plan:
            self.query_one(OutputView).append_output("ui", "meta", "No pending proposal to approve.")
            return
        self.query_one(ChatView).clear_pending_plan()
        self._start_background("iteration", self._controller.approve_plan)

    def action_reject_plan(self) -> None:
        if self._init_busy:
            return
        self._controller.reject_plan()
        self._refresh_from_state()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text.startswith("/"):
            self._handle_local_command(text)
            return

        if self._init_busy:
            self.query_one(OutputView).append_output("ui", "meta", "Startup in progress. Please wait.")
            return
        if self._chat_busy:
            self.query_one(OutputView).append_output("ui", "meta", "Chat already in progress. Send another message in a moment.")
            return

        self._start_background("chat", lambda: self._controller.chat(text))

    def _handle_local_command(self, raw: str) -> None:
        if raw == "/help":
            self.query_one(OutputView).append_output(
                "ui",
                "meta",
                "Commands: /help, /status, /reset. Free text chats with the agent. Y/N approves or rejects pending plans.",
            )
            return
        if raw == "/status":
            state = self._controller.state
            best = "n/a" if state.best_metric is None else f"{state.best_metric:.6f}"
            self.query_one(OutputView).append_output(
                "ui",
                "meta",
                f"state={state.phase} metric={state.metric_name} direction={state.metric_direction} best={best}",
            )
            return
        if raw == "/reset":
            if self._init_busy or self._chat_busy or self._iteration_busy:
                self.query_one(OutputView).append_output("ui", "meta", "Busy: cannot reset during active task.")
                return
            self._controller.reset_history()
            self.query_one(ChatView).clear_pending_plan()
            self.query_one(OutputView).append_output("ui", "meta", "History reset complete.")
            self._refresh_from_state()
            return
        self.query_one(OutputView).append_output("ui", "meta", f"Unknown command: {raw}")

    def _start_background(self, name: str, fn) -> None:
        if name == "initialize":
            if self._init_busy:
                self.query_one(OutputView).append_output("ui", "meta", "Startup already in progress.")
                return
            self._init_busy = True
        elif name == "chat":
            if self._chat_busy:
                self.query_one(OutputView).append_output("ui", "meta", "Chat already in progress.")
                return
            if self._init_busy:
                self.query_one(OutputView).append_output("ui", "meta", "Startup in progress. Please wait.")
                return
            self._chat_busy = True
        elif name == "iteration":
            if self._iteration_busy:
                self.query_one(OutputView).append_output("ui", "meta", "An iteration is already running.")
                return
            if self._init_busy:
                self.query_one(OutputView).append_output("ui", "meta", "Startup in progress. Please wait.")
                return
            self._iteration_busy = True
        else:
            self.query_one(OutputView).append_output("ui", "meta", f"Unknown task: {name}")
            return
        self._refresh_status_bar()

        def _target() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self._push_event(ErrorEvent(message=f"{name} failed", detail=str(exc)))
            finally:
                self._push_event(StartupEvent(stage="task_done", message=name))

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

    def _push_event(self, event: EngineEvent) -> None:
        self._event_queue.put(event)

    def _drain_events(self) -> None:
        processed = 0
        max_per_tick = 200
        while processed < max_per_tick:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            self._handle_event(event)

    def _handle_event(self, event: EngineEvent) -> None:
        output = self.query_one(OutputView)
        chat = self.query_one(ChatView)

        if isinstance(event, StartupEvent):
            if event.stage == "task_done":
                if event.message == "initialize":
                    self._init_busy = False
                elif event.message == "chat":
                    self._chat_busy = False
                elif event.message == "iteration":
                    self._iteration_busy = False
                self._refresh_from_state()
                self._refresh_status_bar()
                return
            output.append_output("startup", "meta", event.message)
            if event.stage == "ready":
                self._refresh_from_state()
            self._refresh_status_bar()
            return

        if isinstance(event, DiscoveryEvent):
            output.append_output("discovery", "meta", event.message)
            self._refresh_status_bar()
            return

        if isinstance(event, OutputLineEvent):
            output.append_output(event.phase, event.stream, event.line)
            return

        if isinstance(event, ChatEvent):
            chat.add_message(event.role, event.content)
            self._refresh_status_bar()
            return

        if isinstance(event, ChatStreamEvent):
            if event.mode == "start":
                chat.start_stream(event.role)
            elif event.mode == "delta":
                chat.append_stream(event.role, event.text)
            else:
                chat.end_stream(event.role, final_text=event.text or None)
            self._refresh_status_bar()
            return

        if isinstance(event, ProposalEvent):
            pending = self._controller.state.pending_action
            if pending is not None:
                chat.set_pending_plan(event.title, pending.plan)
            self._refresh_status_bar()
            return

        if isinstance(event, IterationEvent):
            output.append_output("iteration", "meta", event.message)
            metrics = self.query_one(MetricsView)
            metrics.set_run_context(
                iteration=self._controller.state.running_iteration,
                commit=self._controller.state.running_commit,
                phase=self._controller.state.running_phase,
            )
            self._refresh_from_state()
            return

        if isinstance(event, MetricEvent):
            metrics = self.query_one(MetricsView)
            if event.value is not None:
                metrics.push_value(event.value)
            metrics.update_best(event.best)
            self._refresh_status_bar()
            return

        if isinstance(event, TelemetryEvent):
            metrics = self.query_one(MetricsView)
            metrics.record_metrics(event.metrics)
            self._refresh_status_bar()
            return

        if isinstance(event, DecisionEvent):
            metric_text = "n/a" if event.metric is None else f"{event.metric:.6f}"
            best_text = "n/a" if event.best is None else f"{event.best:.6f}"
            output.append_output(
                "decision",
                "meta",
                f"iter={event.iteration} status={event.status} metric={metric_text} best={best_text} commit={event.commit}",
            )
            self._refresh_from_state()
            return

        if isinstance(event, UsageEvent):
            metrics = self.query_one(MetricsView)
            state = self._controller.state
            metrics.set_token_usage(
                input_total=state.token_input_total,
                output_total=state.token_output_total,
                total=state.token_total,
                last_total=state.last_total_tokens,
                source=state.last_token_source,
            )
            self._refresh_status_bar()
            return

        if isinstance(event, ErrorEvent):
            detail = f" ({event.detail})" if event.detail else ""
            output.append_output("error", "stderr", f"{event.message}{detail}")
            chat.add_message("system", f"{event.message}{detail}")
            self._refresh_status_bar()
            return

    def _refresh_from_state(self) -> None:
        state = self._controller.state
        header = self.query_one("#header", Static)
        best_text = "n/a" if state.best_metric is None else f"{state.best_metric:.6f}"
        active = self._active_modes_text()
        phase_text = state.phase if not active else f"{state.phase}/{active}"
        header.update(
            f"TensorClaw | {state.project_root} | metric={state.metric_name} ({state.metric_direction}) | best={best_text} | phase={phase_text}"
        )

        metrics = self.query_one(MetricsView)
        metrics.set_metric_info(
            metric_name=state.metric_name,
            direction=state.metric_direction,
            baseline=state.metric_baseline,
            best=state.best_metric,
            current=state.current_metric,
        )
        metrics.set_series([point.value for point in state.metric_series])
        metrics.set_aux_series(state.telemetry_series)
        metrics.set_run_context(
            iteration=state.running_iteration,
            commit=state.running_commit,
            phase=state.running_phase,
        )
        metrics.set_token_usage(
            input_total=state.token_input_total,
            output_total=state.token_output_total,
            total=state.token_total,
            last_total=state.last_total_tokens,
            source=state.last_token_source,
        )

        self.query_one(IterationsView).set_iterations(state.iteration_records)

        chat = self.query_one(ChatView)
        if not self._rendered_history:
            for message in state.chat_history:
                chat.add_message(message.role, message.content)
            self._rendered_history = True

        if state.has_pending_plan:
            pending = state.pending_action
            if pending is not None:
                chat.set_pending_plan(f"Plan for iteration {state.current_iteration + 1}", pending.plan)
        else:
            chat.clear_pending_plan()

        self._refresh_status_bar()

    def _active_modes_text(self) -> str:
        active: list[str] = []
        if self._init_busy:
            active.append("startup")
        if self._iteration_busy:
            active.append("run")
        if self._chat_busy:
            active.append("chat")
        return "+".join(active)

    def _refresh_status_bar(self) -> None:
        state = self._controller.state
        live_order = ["step", "train_loss", "val_loss", "val_bpb", "tok_per_sec", "grad_norm"]
        chunks: list[str] = []
        for key in live_order:
            value = state.live_monitor.get(key)
            if value is None:
                continue
            if key == "step":
                chunks.append(f"step={int(value)}")
            elif abs(value) >= 1000:
                chunks.append(f"{key}={value:,.0f}")
            else:
                chunks.append(f"{key}={value:.4g}")
        live_text = ", ".join(chunks[:4])
        active = self._active_modes_text()
        busy = bool(active)
        token_text = f"pi_tokens={state.token_total}"
        self.query_one(StatusBar).update_status(
            phase=state.phase if not active else f"{state.phase}/{active}",
            metric_name=state.metric_name or "n/a",
            direction=state.metric_direction,
            best=state.best_metric,
            iteration=state.current_iteration,
            has_pending_plan=state.has_pending_plan,
            busy=busy,
            monitor=live_text,
            tokens=token_text,
        )
