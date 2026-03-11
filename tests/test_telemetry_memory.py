from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tensorclaw.engine.runner import _parse_live_metrics
from tensorclaw.memory import MemoryEntry, append_memory_entry, read_memory_entries, retrieve_relevant
from tensorclaw.telemetry import MetricSample, append_metric_sample, build_series, read_metric_samples


class TelemetryMemoryTests(unittest.TestCase):
    def test_parse_step_line_metrics(self) -> None:
        line = (
            "step 00042 (38.7%) | loss: 2.918731 | lrm: 0.42 | dt: 312ms | "
            "tok/sec: 6,350 | epoch: 1 | remaining: 183s"
        )
        parsed = _parse_live_metrics(line)
        self.assertEqual(int(parsed["step"]), 42)
        self.assertAlmostEqual(parsed["progress_pct"], 38.7, places=3)
        self.assertAlmostEqual(parsed["train_loss"], 2.918731, places=6)
        self.assertAlmostEqual(parsed["tok_per_sec"], 6350.0, places=3)

    def test_telemetry_and_memory_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "metrics.jsonl"
            memory_path = root / "memory.jsonl"

            append_metric_sample(
                str(metrics_path),
                MetricSample(
                    timestamp_utc="",
                    iteration=1,
                    phase="experiment",
                    metrics={"train_loss": 3.0, "tok_per_sec": 5100},
                    raw_line="",
                ),
            )
            append_metric_sample(
                str(metrics_path),
                MetricSample(
                    timestamp_utc="",
                    iteration=1,
                    phase="experiment",
                    metrics={"train_loss": 2.7, "val_loss": 2.9},
                    raw_line="",
                ),
            )
            samples = read_metric_samples(str(metrics_path))
            series, latest = build_series(samples)
            self.assertEqual(len(series["train_loss"]), 2)
            self.assertAlmostEqual(latest["train_loss"], 2.7, places=6)
            self.assertAlmostEqual(latest["tok_per_sec"], 5100.0, places=6)

            append_memory_entry(
                str(memory_path),
                MemoryEntry(
                    timestamp_utc="",
                    iteration=1,
                    status="discard",
                    metric_name="val_bpb",
                    metric_value=2.4,
                    commit="abc1234",
                    summary="weight decay increase regressed validation",
                    tags=["weight_decay", "regressed", "validation"],
                ),
            )
            append_memory_entry(
                str(memory_path),
                MemoryEntry(
                    timestamp_utc="",
                    iteration=2,
                    status="keep",
                    metric_name="val_bpb",
                    metric_value=2.1,
                    commit="def5678",
                    summary="lr schedule warmdown improved val_bpb",
                    tags=["lr", "warmdown", "improved", "val_bpb"],
                ),
            )
            entries = read_memory_entries(str(memory_path))
            selected = retrieve_relevant(entries, "try lr warmdown again", limit=1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].iteration, 2)


if __name__ == "__main__":
    unittest.main()
