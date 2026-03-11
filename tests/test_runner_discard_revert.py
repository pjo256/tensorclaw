from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tensorclaw.engine.events import DecisionEvent, OutputLineEvent
from tensorclaw.engine.runner import IterationRunner
from tensorclaw.spec import AgentSpec, CommandSpec, GitSpec, LoopSpec, MetricSpec, PathsSpec, ResearchSpec


class RunnerDiscardRevertTests(unittest.TestCase):
    def test_discard_triggers_reset_and_emits_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
            (root / "train.py").write_text("print('hello')\n", encoding="utf-8")
            subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            anchor = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()

            dot = root / ".tensorclaw"
            spec = ResearchSpec(
                version=1,
                name="test-run",
                project_root=str(root),
                target_files=["train.py"],
                commands=CommandSpec(experiment="python3 -c \"print('val_bpb: 2.5')\"", setup=[]),
                metric=MetricSpec(name="val_bpb", direction="minimize", pattern=r"^val_bpb:\s*([-+]?\d*\.?\d+)", baseline=None, min_delta=0.0),
                constraints=[],
                memory=None,
                agent=AgentSpec(enabled=False, command="", proposal_enabled=False),
                git=GitSpec(enabled=True, auto_commit=False, discard_strategy="hard-reset", revert_on_crash=True),
                paths=PathsSpec(
                    results_tsv=str(dot / "results.tsv"),
                    journal_md=str(dot / "journal.md"),
                    logs_dir=str(dot / "logs"),
                    instructions_dir=str(dot / "instructions"),
                ),
                loop=LoopSpec(max_iterations=1, timeout_seconds=60),
                ideas=["test idea"],
                spec_path="",
            )

            events = []
            runner = IterationRunner()

            with patch("tensorclaw.engine.runner.reset_hard") as mock_reset:
                outcome = runner.run_once(
                    spec=spec,
                    current_best=1.5,
                    anchor_commit=anchor,
                    idea="test idea",
                    approved_plan_text="Hypothesis: test",
                    dry_run=False,
                    emit=events.append,
                )

            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.status, "discard")
            mock_reset.assert_called_once_with(spec.project_root, anchor)

            decision_events = [evt for evt in events if isinstance(evt, DecisionEvent)]
            self.assertTrue(decision_events)
            self.assertEqual(decision_events[-1].status, "discard")

            output_events = [evt for evt in events if isinstance(evt, OutputLineEvent)]
            self.assertTrue(any(evt.phase == "experiment" for evt in output_events))


if __name__ == "__main__":
    unittest.main()
