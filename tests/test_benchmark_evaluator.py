from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from evals.benchmark.evaluator import _experiment_metadata, _require_cuda, evaluate
from evals.benchmark.metrics import aggregate_metrics, task_metrics
from model.local import LocalTransformersClient


class BenchmarkMetricsTests(unittest.TestCase):
    def test_task_metrics_count_valid_invalid_calls_and_tokens(self) -> None:
        record = {
            "success_condition": {"required_tools": ["grep"]},
            "success": False,
            "failure_category": "invalid_tool_call",
            "latency_seconds": 3.5,
            "token_usage": {"prompt_tokens": 90, "generated_tokens": 10, "total_tokens": 100},
            "trajectory": {
                "events": [
                    {"kind": "model", "model_output": '<tool_call>{"name":"grep"}</tool_call>'},
                    {"kind": "tool", "tool_name": "grep", "executed": True},
                    {"kind": "model", "model_output": '<tool_call>{bad}</tool_call>', "error": "invalid JSON"},
                ]
            },
        }
        metrics = task_metrics(record)
        self.assertEqual(metrics["tool_calls_attempted"], 2)
        self.assertEqual(metrics["tool_calls_schema_valid"], 1)
        self.assertEqual(metrics["tool_calls_invalid"], 1)
        self.assertEqual(metrics["argument_schema_validity"], 0.5)
        self.assertEqual(metrics["invalid_tool_call_rate"], 0.5)
        self.assertEqual(metrics["tool_selection_accuracy"], 1.0)
        self.assertEqual(metrics["total_tokens"], 100)

    def test_aggregate_metrics_include_success_latency_tokens_and_failures(self) -> None:
        rows = [
            {
                "success": True,
                "failure_category": "none",
                "latency_seconds": 2.0,
                "success_condition": {"required_tools": ["grep"]},
                "token_usage": {"total_tokens": 50},
                "trajectory": {"events": [
                    {"kind": "model", "model_output": '<tool_call>{}</tool_call>'},
                    {"kind": "tool", "tool_name": "grep", "executed": True},
                    {"kind": "model", "model_output": "done"},
                ]},
            },
            {
                "success": False,
                "failure_category": "validation_failure",
                "latency_seconds": 4.0,
                "success_condition": {"required_tools": ["shell"]},
                "token_usage": None,
                "trajectory": {"events": []},
            },
        ]
        summary = aggregate_metrics(rows)
        self.assertEqual(summary["task_success_rate"], 0.5)
        self.assertEqual(summary["average_latency_seconds"], 3.0)
        self.assertEqual(summary["average_tokens_per_task"], 50)
        self.assertEqual(summary["token_metric_tasks"], 1)
        self.assertEqual(summary["failure_categories"], {"validation_failure": 1})


class BenchmarkEvaluatorSplitTests(unittest.TestCase):
    def test_training_support_run_rejects_heldout_id_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "not in split train-support"):
                evaluate(
                    split="train-support",
                    model_path="unused",
                    output_path=root / "out.jsonl",
                    summary_path=root / "summary.md",
                    task_ids=["B005"],
                )
            self.assertFalse((root / "out.jsonl").exists())

    def test_cuda_mode_stops_before_creating_results_when_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("evals.benchmark.evaluator._require_cuda", side_effect=RuntimeError("no CUDA device")):
                with self.assertRaisesRegex(RuntimeError, "no CUDA device"):
                    evaluate(
                        split="held-out-eval",
                        model_path="unused",
                        output_path=root / "out.jsonl",
                        summary_path=root / "summary.md",
                        device="cuda",
                    )
            self.assertFalse((root / "out.jsonl").exists())

    def test_experiment_metadata_captures_model_prompt_schemas_hardware_and_split_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "config.json").write_text(
                json.dumps({"_name_or_path": "Qwen/Qwen3-1.7B", "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
                encoding="utf-8",
            )
            client = LocalTransformersClient(str(model_path), device="cuda")
            metadata = _experiment_metadata(
                split="held-out-eval",
                model_path=str(model_path),
                device="cuda",
                client=client,
                max_steps=16,
                max_seconds=600,
                validation_timeout=120,
                hardware={"gpus": [{"name": "GTX 1060"}]},
            )
            self.assertEqual(metadata["model"]["model_type"], "qwen3")
            self.assertEqual(metadata["model"]["inference_configuration"]["quantization"], "none")
            self.assertTrue(metadata["prompt"]["sha256"])
            self.assertEqual(len(metadata["tool_schemas"]["definitions"]), 5)
            self.assertEqual(metadata["benchmark"]["split"], "held-out-eval")
            self.assertEqual(len(metadata["benchmark"]["split_task_ids"]), 10)
            self.assertEqual(metadata["hardware"]["gpus"][0]["name"], "GTX 1060")

    def test_cuda_requirement_rejects_runtime_without_device(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def device_count():
                return 0

        class FakeTorch:
            cuda = FakeCuda()

        with self.assertRaisesRegex(RuntimeError, "cannot access a CUDA device"):
            _require_cuda(FakeTorch())


if __name__ == "__main__":
    unittest.main()
