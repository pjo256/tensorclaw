from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tensorclaw.engine.controller import ResearchController
from tensorclaw.engine.events import ChatEvent
from tensorclaw.engine.models import PendingAction, ProposalPlan


class ControllerPlanFlowTests(unittest.TestCase):
    def test_reject_plan_clears_pending_and_emits_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            controller = ResearchController(project_root=Path(tmp), dry_run=True, event_sink=events.append)
            controller.state.pending_action = PendingAction(
                type="proposal",
                plan=ProposalPlan(
                    hypothesis="Try a single lr change",
                    planned_edits=["train.py: tweak lr"],
                    expected_impact="Lower val_bpb",
                    risk="Could regress",
                ),
            )

            controller.reject_plan()
            self.assertIsNone(controller.state.pending_action)
            chat_events = [evt for evt in events if isinstance(evt, ChatEvent)]
            self.assertTrue(chat_events)
            self.assertIn("rejected", chat_events[-1].content.lower())

    def test_approve_plan_calls_run_iteration_with_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = ResearchController(project_root=Path(tmp), dry_run=True, event_sink=None)
            controller.state.pending_action = PendingAction(
                type="proposal",
                plan=ProposalPlan(
                    hypothesis="Tune warmup schedule",
                    planned_edits=["train.py: warmup ratio"],
                    expected_impact="Lower val_bpb",
                    risk="Could underfit",
                    direction="Run scheduler ablation",
                ),
            )

            captured: dict[str, object] = {}

            def fake_run_iteration(direction: str | None = None, approved_plan: ProposalPlan | None = None) -> None:
                captured["direction"] = direction
                captured["approved_plan"] = approved_plan

            controller.run_iteration = fake_run_iteration  # type: ignore[assignment]
            controller.approve_plan()

            self.assertIsNone(controller.state.pending_action)
            self.assertEqual(captured.get("direction"), "Run scheduler ablation")
            self.assertIsInstance(captured.get("approved_plan"), ProposalPlan)


if __name__ == "__main__":
    unittest.main()
