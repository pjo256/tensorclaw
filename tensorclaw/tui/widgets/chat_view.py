from __future__ import annotations

from rich.markup import escape
from textual.containers import VerticalScroll
from textual.widgets import Static

from ...engine.models import ProposalPlan


class MessageBubble(Static):
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self._bubble_text = content
        super().__init__(self._render_text(), classes=f"message-bubble role-{role}")

    def set_content(self, content: str) -> None:
        self._bubble_text = content
        self.update(self._render_text())

    def _render_text(self) -> str:
        role_label = {"user": "You", "assistant": "Agent", "system": "TensorClaw"}.get(self.role, self.role.title())
        return f"[b]{role_label}:[/b] {escape(self._bubble_text)}"


class ProposalCard(Static):
    def __init__(self, title: str, plan: ProposalPlan) -> None:
        super().__init__(self._build_body(title, plan), classes="proposal-card")

    def update_plan(self, title: str, plan: ProposalPlan) -> None:
        self.update(self._build_body(title, plan))

    @staticmethod
    def _build_body(title: str, plan: ProposalPlan) -> str:
        edits = "\n".join(f"- {escape(item)}" for item in plan.planned_edits) if plan.planned_edits else "- (none)"
        return (
            f"[b]{escape(title)}[/b]\n"
            f"Hypothesis: {escape(plan.hypothesis)}\n\n"
            f"Planned edits:\n{edits}\n\n"
            f"Expected impact: {escape(plan.expected_impact)}\n"
            f"Risk: {escape(plan.risk)}\n\n"
            "[b][Y][/b] run  [b][N][/b] cancel"
        )


class ChatView(VerticalScroll):
    def __init__(self) -> None:
        super().__init__(id="chat-pane")
        self._proposal_card: ProposalCard | None = None
        self._stream_bubble: MessageBubble | None = None
        self._stream_role: str | None = None
        self._stream_text: str = ""

    def add_message(self, role: str, content: str) -> None:
        if self._stream_bubble is not None and self._stream_role == role:
            self.end_stream(role=role, final_text=content)
            return
        self.mount(MessageBubble(role=role, content=content))
        self.call_after_refresh(self.scroll_end, animate=False)

    def start_stream(self, role: str) -> None:
        if self._stream_bubble is not None and self._stream_role != role:
            self.end_stream(role=self._stream_role or role)
        if self._stream_bubble is None:
            bubble = MessageBubble(role=role, content="")
            self.mount(bubble)
            self._stream_bubble = bubble
            self._stream_role = role
            self._stream_text = ""
        self.call_after_refresh(self.scroll_end, animate=False)

    def append_stream(self, role: str, text: str) -> None:
        if not text:
            return
        if self._stream_bubble is None or self._stream_role != role:
            self.start_stream(role)
        self._stream_text += text
        if self._stream_bubble is not None:
            self._stream_bubble.set_content(self._stream_text)
        self.call_after_refresh(self.scroll_end, animate=False)

    def end_stream(self, role: str, final_text: str | None = None) -> None:
        if self._stream_bubble is None:
            if final_text:
                self.add_message(role=role, content=final_text)
            return
        if final_text is not None:
            self._stream_text = final_text
            self._stream_bubble.set_content(final_text)
        self._stream_bubble = None
        self._stream_role = None
        self._stream_text = ""
        self.call_after_refresh(self.scroll_end, animate=False)

    def set_pending_plan(self, title: str, plan: ProposalPlan) -> None:
        if self._proposal_card is None:
            self._proposal_card = ProposalCard(title=title, plan=plan)
            self.mount(self._proposal_card)
        else:
            self._proposal_card.update_plan(title=title, plan=plan)
        self.call_after_refresh(self.scroll_end, animate=False)

    def clear_pending_plan(self) -> None:
        if self._proposal_card is not None:
            self._proposal_card.remove()
            self._proposal_card = None

    def clear_stream(self) -> None:
        self._stream_bubble = None
        self._stream_role = None
        self._stream_text = ""
