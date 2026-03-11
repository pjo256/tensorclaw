from __future__ import annotations

import unittest

from tensorclaw.engine.controller import _parse_pending_payload


class PlanConfirmationTests(unittest.TestCase):
    def test_parse_pending_payload_approve(self) -> None:
        reply, action = _parse_pending_payload(
            {
                "assistant_reply": "Sounds good, I'll run it now.",
                "action": "approve",
            }
        )
        self.assertEqual(reply, "Sounds good, I'll run it now.")
        self.assertEqual(action, "approve")

    def test_parse_pending_payload_reject(self) -> None:
        reply, action = _parse_pending_payload(
            {
                "assistant_reply": "Okay, I'll cancel that plan.",
                "action": "reject",
            }
        )
        self.assertEqual(reply, "Okay, I'll cancel that plan.")
        self.assertEqual(action, "reject")

    def test_parse_pending_payload_invalid_action(self) -> None:
        _, action = _parse_pending_payload(
            {
                "assistant_reply": "Let's discuss first.",
                "action": "run_now",
            }
        )
        self.assertEqual(action, "chat_only")


if __name__ == "__main__":
    unittest.main()
