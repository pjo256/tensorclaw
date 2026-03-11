from __future__ import annotations

import unittest

from tensorclaw.engine.controller import (
    _extract_assistant_reply_candidate,
    _extract_stream_delta_from_json_line,
    _extract_usage_from_json_line,
    _parse_chat_payload,
    _parse_proposal_from_text,
)


class ChatIntentTests(unittest.TestCase):
    def test_question_maps_to_chat_only(self) -> None:
        payload = {
            "assistant_reply": "It comes from train.py output.",
            "intent": "chat_only",
        }
        reply, intent, proposal = _parse_chat_payload(payload)
        self.assertEqual(reply, "It comes from train.py output.")
        self.assertEqual(intent, "chat_only")
        self.assertIsNone(proposal)

    def test_proposal_payload_maps_to_plan(self) -> None:
        payload = {
            "assistant_reply": "I have a plan.",
            "intent": "proposal_ready",
            "proposal": {
                "hypothesis": "Tune LR schedule.",
                "planned_edits": ["train.py: adjust warmup"],
                "expected_impact": "Lower val_bpb.",
                "risk": "Could underfit.",
                "direction": "Try a softer warmdown",
            },
        }
        _, intent, proposal = _parse_chat_payload(payload)
        self.assertEqual(intent, "proposal_ready")
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.hypothesis, "Tune LR schedule.")
        self.assertEqual(proposal.direction, "Try a softer warmdown")

    def test_invalid_intent_falls_back_to_chat_only(self) -> None:
        payload = {
            "assistant_reply": "Something",
            "intent": "run_it",
        }
        _, intent, proposal = _parse_chat_payload(payload)
        self.assertEqual(intent, "chat_only")
        self.assertIsNone(proposal)

    def test_plain_text_proposal_parser(self) -> None:
        text = """Plan: Tune scheduler constants.
- train.py: increase warmup ratio
Impact: improve val_bpb.
Risk: could slow early learning.
"""
        plan = _parse_proposal_from_text(text)
        self.assertIn("scheduler", plan.hypothesis.lower())
        self.assertEqual(len(plan.planned_edits), 1)
        self.assertIn("val_bpb", plan.expected_impact)

    def test_stream_delta_extraction_openai_style(self) -> None:
        line = '{"type":"response.output_text.delta","delta":"hello "}'
        self.assertEqual(_extract_stream_delta_from_json_line(line), "hello ")

    def test_stream_delta_extraction_nested_style(self) -> None:
        line = '{"type":"assistant_delta","delta":{"content":[{"type":"text","text":"world"}]}}'
        self.assertEqual(_extract_stream_delta_from_json_line(line), "world")

    def test_stream_delta_extraction_message_update_style(self) -> None:
        line = (
            '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"hello"}}'
        )
        self.assertEqual(_extract_stream_delta_from_json_line(line), "hello")

    def test_extract_assistant_reply_candidate_partial(self) -> None:
        partial = '{"assistant_reply":"Hel'
        self.assertEqual(_extract_assistant_reply_candidate(partial), "Hel")

    def test_extract_assistant_reply_candidate_complete(self) -> None:
        payload = '{"assistant_reply":"Hello!","intent":"chat_only"}'
        self.assertEqual(_extract_assistant_reply_candidate(payload), "Hello!")

    def test_extract_usage_from_json_line_usage_container(self) -> None:
        line = '{"type":"turn_end","usage":{"input_tokens":120,"output_tokens":40,"total_tokens":160}}'
        usage = _extract_usage_from_json_line(line)
        self.assertEqual(usage.get("input"), 120)
        self.assertEqual(usage.get("output"), 40)
        self.assertEqual(usage.get("total"), 160)

    def test_extract_usage_from_json_line_prompt_completion_fallback(self) -> None:
        line = '{"prompt_tokens":33,"completion_tokens":7}'
        usage = _extract_usage_from_json_line(line)
        self.assertEqual(usage.get("input"), 33)
        self.assertEqual(usage.get("output"), 7)
        self.assertEqual(usage.get("total"), 40)


if __name__ == "__main__":
    unittest.main()
