from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.definitions import get_tool_schemas
from training.agenttune import hf_gpu_smoke
from training.agenttune.support_prompt import SUPPORT_SYSTEM_PROMPT


class RepeatRecoveryPolicyTests(unittest.TestCase):
    def _policy(self, retry_outputs: list[str]) -> hf_gpu_smoke.TinyAyaPolicy:
        policy = hf_gpu_smoke.TinyAyaPolicy(None, None, None, get_tool_schemas())
        previous = {"tool_name": "read_file", "parameters": {
            "path": "shop/config.py", "start_line": 1, "end_line": 20,
        }}
        previous_record = policy._action_validity(previous)
        policy.selected_actions.append(previous_record)
        policy.raw_outputs.append(json.dumps([previous]))
        policy.assistant_turns.append(json.dumps([previous]))
        policy.parse_errors.append(None)
        policy.ignored_trailing_text.append("")
        policy.response_adaptations.append(None)
        policy.prompts.append([{"role": "system", "content": SUPPORT_SYSTEM_PROMPT}])

        outputs = iter(retry_outputs)

        def fake_generate(messages: list[dict[str, str]]) -> str:
            policy.prompts.append([dict(message) for message in messages])
            output = next(outputs)
            policy.raw_outputs.append(output)
            policy.prompt_tokens_seen += 10
            policy.generated_tokens_seen += 5
            return output

        policy._generate_raw = fake_generate  # type: ignore[method-assign]
        return policy

    @staticmethod
    def _state() -> SimpleNamespace:
        return SimpleNamespace(
            step=2,
            events=[
                SimpleNamespace(kind="observation", payload={"text": "Task: report the setting."}),
                SimpleNamespace(kind="observation", payload={
                    "text": 'Tool result: {"success": true, "stdout": "DEFAULT_TIMEOUT_SECONDS = 10"}'
                }),
            ],
        )

    def test_duplicate_call_gets_one_finish_only_model_retry(self) -> None:
        duplicate = {"tool_name": "read_file", "parameters": {
            "path": "shop/config.py", "start_line": 1, "end_line": 20,
        }}
        finish = {"tool_name": "finish", "parameters": {"answer": "The default timeout is 10 seconds."}}
        policy = self._policy([json.dumps([duplicate]), json.dumps([finish])])

        with patch.dict(hf_gpu_smoke.EVAL_CONFIG, {
            "prompt_variant": "support-v5-repeat-finish-retry",
            "prompt": {"system": SUPPORT_SYSTEM_PROMPT},
        }):
            result = policy(self._state())

        self.assertEqual(result, {"name": "finish", "arguments": finish["parameters"]})
        self.assertEqual(policy.final_response_source, "model")
        self.assertEqual(policy.returned_action_output_indices, [2])
        self.assertEqual(policy.response_adaptations[-2:], [
            "repeated_action_finish_retry_triggered",
            "repeated_action_finish_retry_succeeded",
        ])
        self.assertIn("Tools are disabled", policy.prompts[-1][-1]["content"])
        self.assertEqual(policy.selected_actions[-1]["name"], "finish")

    def test_non_finish_recovery_is_rejected_without_harness_answer_text(self) -> None:
        duplicate = {"tool_name": "read_file", "parameters": {
            "path": "shop/config.py", "start_line": 1, "end_line": 20,
        }}
        repeated_again = {"tool_name": "read_file", "parameters": {
            "path": "shop/config.py", "start_line": 1, "end_line": 20,
        }}
        policy = self._policy([json.dumps([duplicate]), json.dumps([repeated_again])])

        with patch.dict(hf_gpu_smoke.EVAL_CONFIG, {
            "prompt_variant": "support-v5-repeat-finish-retry",
            "prompt": {"system": SUPPORT_SYSTEM_PROMPT},
        }):
            result = policy(self._state())

        self.assertEqual(result, {"name": "finish", "arguments": {"answer": ""}})
        self.assertEqual(policy.final_response_source, "harness_empty_stop")
        self.assertEqual(policy.returned_action_output_indices, [None])
        self.assertEqual(policy.response_adaptations[-1], "repeated_action_finish_retry_failed")
        self.assertNotIn("I could not complete the task", json.dumps(policy.selected_actions))
        self.assertFalse(policy.selected_actions[-2]["schema_valid"])


if __name__ == "__main__":
    unittest.main()
