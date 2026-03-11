from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tensorclaw.engine.controller import ResearchController
from tensorclaw.ledger import LedgerRow, append_row, ensure_ledger
from tensorclaw.spec import AgentSpec, CommandSpec, GitSpec, LoopSpec, MetricSpec, PathsSpec, ResearchSpec


class HistoryLoadingTests(unittest.TestCase):
    def test_refresh_iteration_state_loads_all_rows_and_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dot = root / ".tensorclaw"
            tsv = dot / "results.tsv"
            ensure_ledger(str(tsv))
            append_row(
                str(tsv),
                LedgerRow(
                    iteration=1,
                    commit="aaaa111",
                    metric_name="val_bpb",
                    metric_value=2.1,
                    memory_gb=0.0,
                    status="keep",
                    description="baseline run",
                ),
            )
            append_row(
                str(tsv),
                LedgerRow(
                    iteration=2,
                    commit="bbbb222",
                    metric_name="val_bpb",
                    metric_value=2.3,
                    memory_gb=0.0,
                    status="discard",
                    description="regressed run",
                ),
            )
            append_row(
                str(tsv),
                LedgerRow(
                    iteration=3,
                    commit="cccc333",
                    metric_name="val_bpb",
                    metric_value=1.9,
                    memory_gb=0.0,
                    status="keep",
                    description="improved run",
                ),
            )

            controller = ResearchController(project_root=root, dry_run=False, event_sink=None)
            controller.state.spec = ResearchSpec(
                version=1,
                name="test",
                project_root=str(root),
                target_files=["train.py"],
                commands=CommandSpec(experiment="python3 -c \"print('val_bpb: 1.0')\"", setup=[]),
                metric=MetricSpec(name="val_bpb", direction="minimize", pattern=r"^val_bpb:\s*([-+]?\d*\.?\d+)", baseline=None, min_delta=0.0),
                constraints=[],
                memory=None,
                agent=AgentSpec(enabled=False, command="", proposal_enabled=False),
                git=GitSpec(enabled=False, auto_commit=False),
                paths=PathsSpec(
                    results_tsv=str(tsv),
                    journal_md=str(dot / "journal.md"),
                    logs_dir=str(dot / "logs"),
                    instructions_dir=str(dot / "instructions"),
                ),
                loop=LoopSpec(max_iterations=1, timeout_seconds=60),
                ideas=["idea"],
                spec_path="",
            )

            controller._refresh_iteration_state()  # noqa: SLF001
            records = controller.state.iteration_records
            self.assertEqual(len(records), 3)
            self.assertEqual(controller.state.best_metric, 1.9)
            self.assertEqual(records[-1].iteration, 3)
            self.assertTrue(records[-1].is_best)
            self.assertTrue(records[-1].is_latest)


if __name__ == "__main__":
    unittest.main()
