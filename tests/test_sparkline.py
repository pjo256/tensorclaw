from __future__ import annotations

import unittest

from tensorclaw.tui.widgets.metrics_view import Sparkline


class SparklineTests(unittest.TestCase):
    def test_empty_values(self) -> None:
        self.assertEqual(Sparkline.render([]), "")

    def test_constant_values(self) -> None:
        values = [2.0] * 6
        rendered = Sparkline.render(values)
        self.assertEqual(len(rendered), len(values))
        self.assertEqual(len(set(rendered)), 1)

    def test_width_sampling(self) -> None:
        values = [float(i) for i in range(100)]
        rendered = Sparkline.render(values, width=20)
        self.assertEqual(len(rendered), 20)


if __name__ == "__main__":
    unittest.main()
