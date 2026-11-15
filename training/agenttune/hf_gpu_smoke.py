"""One real Tiny Aya + AgentTune tool-use smoke task for HF GPU Jobs.

This file is staged with only the Terminal-SLM tool layer and B006's
train-support snapshot. It deliberately does not run the formal evaluator or
touch any held-out task.
"""

from __future__ import annotations

import importlib.metadata
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from shutil import copytree
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = Path(os.environ.get(
    "TERMINAL_SLM_CONFIG",
    str(PROJECT_ROOT / "configs" / "tiny_aya_eval.json"),
)).resolve()
EVAL_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_ID = EVAL_CONFIG["model"]["repo_id"]
MODEL_REVISION = EVAL_CONFIG["model"]["revision"]
TASK_ID = "B002"  # train-support only: locate the route table using grep
MAX_STEPS = EVAL_CONFIG["agent"]["max_steps"]
MAX_NEW_TOKENS = EVAL_CONFIG["generation"]["max_new_tokens"]
PURPOSE_TERMS = ("maps", "method", "path", "route")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_frozen_config(schemas: list[dict[str, Any]]) -> None:
    hashes = EVAL_CONFIG["hashes"]
    if _canonical_sha256(EVAL_CONFIG["prompt"]["system"]) != hashes["prompt_sha256"]:
        raise RuntimeError("frozen system prompt hash does not match evaluation config")
    if _canonical_sha256(schemas) != hashes["tool_schemas_sha256"]:
        raise RuntimeError("tool schema hash does not match frozen evaluation config")


def verify_runtime_versions(torch_module: Any) -> dict[str, str | None]:
    actual = {
        "torch": str(torch_module.__version__).split("+", 1)[0],
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "agenttune": importlib.metadata.version("agenttune"),
    }
    expected = EVAL_CONFIG["runtime"]
    mismatches = [f"{name}: expected {expected[name]}, found {actual[name]}" for name in actual if actual[name] != expected[name]]
    if mismatches:
        raise RuntimeError("runtime differs from frozen evaluation config: " + "; ".join(mismatches))
    return actual


class TinyAyaPolicy:
    """Tiny Aya policy using AgentTune's documented Cohere fallback format.

    The format is a JSON list of {tool_name, parameters} objects. Tool
    observations are supplied as user turns, matching AgentTune's Cohere
    fallback because Tiny Aya's chat template does not render tool-role turns.
    """

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        torch: Any,
        schemas: list[dict[str, Any]],
        *,
        time_limit_seconds: float | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch
        self.schemas = schemas
        self.generation_config = EVAL_CONFIG["generation"]
        self.raw_outputs: list[str] = []
        self.assistant_turns: list[str] = []
        self.ignored_trailing_text: list[str] = []
        self.response_adaptations: list[str | None] = []
        self.prompts: list[list[dict[str, str]]] = []
        self.parse_errors: list[str | None] = []
        self.selected_actions: list[dict[str, Any]] = []
        self.returned_action_output_indices: list[int | None] = []
        self.final_response_source: str | None = None
        self.prompt_tokens_seen = 0
        self.generated_tokens_seen = 0
        self.time_limit_seconds = time_limit_seconds
        self.started_at = time.monotonic()

    def _generate_raw(self, messages: list[dict[str, str]]) -> str:
        if self.time_limit_seconds is not None and time.monotonic() - self.started_at >= self.time_limit_seconds:
            raise TimeoutError("per-task model-call time limit reached")
        messages = [dict(message) for message in messages]
        while True:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if inputs["input_ids"].shape[-1] <= self.generation_config["max_prompt_tokens"]:
                break
            # Keep the original user task, system instructions, and the most
            # recent observations; trim the oldest complete assistant/user pair.
            if len(messages) > 4:
                del messages[2:4]
                continue
            raise ValueError("prompt exceeds the frozen max_prompt_tokens limit")
        self.prompts.append(messages)
        inputs = inputs.to(self.model.device)
        prompt_length = inputs["input_ids"].shape[-1]
        self.prompt_tokens_seen += int(prompt_length)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config["max_new_tokens"],
                do_sample=self.generation_config["do_sample"],
                use_cache=self.generation_config["use_cache"],
            )
        self.generated_tokens_seen += int(generated.shape[-1] - prompt_length)
        if self.time_limit_seconds is not None and time.monotonic() - self.started_at >= self.time_limit_seconds:
            raise TimeoutError("model generation exceeded the per-task time limit")
        raw = self.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        self.raw_outputs.append(raw)
        return raw

    def __call__(self, state: Any) -> dict[str, Any]:
        messages = self._messages(state)
        raw = self._generate_raw(messages)
        output_index = len(self.raw_outputs) - 1
        try:
            action, trailing = self._parse_action(raw)
        except Exception as exc:
            self.parse_errors.append(f"{type(exc).__name__}: {exc}")
            raise
        self.parse_errors.append(None)
        adaptation: str | None = None
        prompt_variant = EVAL_CONFIG.get("prompt_variant", "")
        is_support_v2_or_later = prompt_variant in {
            "support-v2-action-evidence", "support-v3-json-observations",
            "support-v4-json-repeat-guard", "support-v5-repeat-finish-retry",
        }
        if is_support_v2_or_later:
            if trailing and re.search(r"\[\s*\{", trailing):
                adaptation = "additional_action_arrays_ignored_after_first"
            elif trailing:
                # Run the selected tool and ask again after the observation;
                # prose must not replace a needed inspection/edit/test action.
                adaptation = "trailing_text_ignored_after_selected_action"
        elif getattr(state, "step", 0) > 0 and trailing and "[" not in trailing:
            # Preserve the frozen baseline's original response adaptation.
            discarded_action = action["tool_name"]
            action = {"tool_name": "finish", "parameters": {"answer": trailing}}
            adaptation = f"plain_text_tail_to_finish; discarded_action={discarded_action}"
        action_record = self._action_validity(action)
        repeated_action_stopped = False
        if prompt_variant == "support-v3-json-observations" and action["tool_name"] in {"grep", "read_file"} and action_record["schema_valid"]:
            action_key = json.dumps(
                {"name": action["tool_name"], "arguments": action["parameters"]},
                sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            )
            repeated_action_stopped = any(
                previous.get("schema_valid")
                and previous.get("name") in {"grep", "read_file"}
                and json.dumps(
                    {"name": previous.get("name"), "arguments": previous.get("arguments", {})},
                    sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                ) == action_key
                for previous in self.selected_actions
            )
        elif prompt_variant in {"support-v4-json-repeat-guard", "support-v5-repeat-finish-retry"} and action["tool_name"] != "finish" and action_record["schema_valid"]:
            action_key = json.dumps(
                {"name": action["tool_name"], "arguments": action["parameters"]},
                sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            )
            for index in range(len(self.selected_actions) - 1, -1, -1):
                previous = self.selected_actions[index]
                previous_key = json.dumps(
                    {"name": previous.get("name"), "arguments": previous.get("arguments", {})},
                    sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                )
                if not previous.get("schema_valid") or previous_key != action_key:
                    continue
                edited_since = any(
                    earlier.get("name") == "write_file" and earlier.get("schema_valid")
                    for earlier in self.selected_actions[index + 1:]
                )
                repeated_action_stopped = not edited_since
                break
        if repeated_action_stopped:
            action_record["blocked_by_loop_guard"] = True
        self.selected_actions.append(action_record)
        if repeated_action_stopped:
            if prompt_variant == "support-v5-repeat-finish-retry":
                self.ignored_trailing_text.append(trailing)
                self.response_adaptations.append("repeated_action_finish_retry_triggered")
                retry_messages = [dict(message) for message in self.prompts[output_index]]
                retry_messages.extend([
                    {"role": "assistant", "content": json.dumps([action], ensure_ascii=False)},
                    {
                        "role": "user",
                        "content": (
                            "Your proposed action repeats a tool call already made, so it was not run again. "
                            "The earlier tool observation is available above. Tools are disabled for this recovery turn. "
                            "Return exactly one JSON action using only tool_name=finish and parameters={\"answer\": "
                            "\"your concise evidence-based answer\"}. If the observations do not support an answer, "
                            "say what is unknown. Do not propose another tool call."
                        ),
                    },
                ])
                retry_raw = self._generate_raw(retry_messages)
                retry_output_index = len(self.raw_outputs) - 1
                try:
                    retry_action, retry_trailing = self._parse_action(retry_raw)
                except Exception as exc:
                    self.parse_errors.append(f"{type(exc).__name__}: {exc}")
                    retry_action = None
                    retry_trailing = ""
                else:
                    self.parse_errors.append(None)

                if retry_action is not None:
                    retry_record = self._action_validity(retry_action)
                    if (
                        retry_action.get("tool_name") != "finish"
                        or not retry_record["schema_valid"]
                        or not str(retry_action["parameters"].get("answer", "")).strip()
                    ):
                        retry_record["schema_valid"] = False
                        retry_record["error"] = "finish-only repeat recovery requires a non-empty finish answer"
                        self.selected_actions.append(retry_record)
                        retry_action = None

                if retry_action is not None:
                    self.selected_actions.append(retry_record)
                    self.returned_action_output_indices.append(retry_output_index)
                    self.assistant_turns.append(json.dumps([retry_action], ensure_ascii=False))
                    self.ignored_trailing_text.append(retry_trailing)
                    self.response_adaptations.append("repeated_action_finish_retry_succeeded")
                    self.final_response_source = "model"
                    return {"name": "finish", "arguments": retry_action["parameters"]}

                self.returned_action_output_indices.append(None)
                self.assistant_turns.append(json.dumps([
                    {"tool_name": "finish", "parameters": {"answer": ""}}
                ], ensure_ascii=False))
                self.ignored_trailing_text.append(retry_trailing)
                self.response_adaptations.append("repeated_action_finish_retry_failed")
                self.final_response_source = "harness_empty_stop"
                self.selected_actions.append({
                    "name": "finish",
                    "arguments": {"answer": ""},
                    "schema_valid": True,
                    "source": "harness",
                })
                return {"name": "finish", "arguments": {"answer": ""}}

            action = {
                "tool_name": "finish",
                "parameters": {"answer": "I could not complete the task: I repeated the same read-only tool call without new evidence."},
            }
            adaptation = (
                "repeated_read_action_stopped"
                if prompt_variant == "support-v3-json-observations"
                else "repeated_action_stopped_without_intervening_edit"
            )
            self.returned_action_output_indices.append(None)
            self.final_response_source = "harness_empty_stop" if prompt_variant == "support-v5-repeat-finish-retry" else "harness"
        elif trailing and is_support_v2_or_later:
            # Run the selected tool and ask again after the observation;
            # prose must not replace a needed inspection/edit/test action.
            adaptation = adaptation or "trailing_text_ignored_after_selected_action"
        self.returned_action_output_indices.append(output_index)
        if action["tool_name"] == "finish":
            self.final_response_source = "model"
        # Only echo the action actually handed to AgentTune. Tiny Aya sometimes
        # emits several proposed actions in one completion; replaying discarded
        # proposals alongside later tool observations corrupts the dialogue.
        self.assistant_turns.append(json.dumps([action], ensure_ascii=False))
        self.ignored_trailing_text.append(trailing)
        self.response_adaptations.append(adaptation)
        return {"name": action["tool_name"], "arguments": action["parameters"]}

    def _messages(self, state: Any) -> list[dict[str, str]]:
        visible_tools = []
        for schema in self.schemas:
            function = schema["function"]
            properties = function["parameters"].get("properties", {})
            signature = ", ".join(
                f"{name}: {spec.get('type', 'value')}"
                + (f" >= {spec['minimum']}" if "minimum" in spec else "")
                + (f" <= {spec['maximum']}" if "maximum" in spec else "")
                + (f" (default {spec['default']})" if "default" in spec else "")
                for name, spec in properties.items()
            )
            visible_tools.append(f"- {function['name']}({signature}): {function['description']}")
        system = EVAL_CONFIG["prompt"]["system"]
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        observations = []
        for event in state.events:
            kind = str(getattr(event.kind, "value", event.kind)).lower()
            payload = event.payload if isinstance(event.payload, dict) else {}
            if "observation" in kind and payload.get("text"):
                value = str(payload["text"])
                if EVAL_CONFIG.get("prompt_variant") in {"support-v3-json-observations", "support-v4-json-repeat-guard", "support-v5-repeat-finish-retry"}:
                    value = self._format_observation(value)
                observations.append(value)
        if observations:
            messages.append({"role": "user", "content": observations[0]})
            for index, observation in enumerate(observations[1:]):
                if index < len(self.raw_outputs):
                    messages.append({"role": "assistant", "content": self.assistant_turns[index]})
                content = observation if observation.startswith(("Tool result:\n", "Tool result (JSON):\n")) else "Tool result:\n" + observation
                messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _format_observation(text: str) -> str:
        prefix = "Tool result:\n"
        raw_value = text[len(prefix):] if text.startswith(prefix) else text
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return text
        if not isinstance(value, (dict, list)):
            return text
        return "Tool result (JSON):\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str)

    def _action_validity(self, action: dict[str, Any]) -> dict[str, Any]:
        name = action["tool_name"]
        arguments = action["parameters"]
        if name == "finish":
            valid = set(arguments) == {"answer"} and isinstance(arguments.get("answer"), str)
            return {"name": name, "arguments": arguments, "schema_valid": valid}
        known = {schema["function"]["name"] for schema in self.schemas}
        if name not in known:
            return {"name": name, "arguments": arguments, "schema_valid": False, "error": "unknown tool"}
        try:
            from tools.definitions import validate_tool_arguments
            validate_tool_arguments(name, arguments)
        except (TypeError, ValueError) as exc:
            return {"name": name, "arguments": arguments, "schema_valid": False, "error": str(exc)}
        return {"name": name, "arguments": arguments, "schema_valid": True}

    @staticmethod
    def _parse_action(text: str) -> tuple[dict[str, Any], str]:
        decoder = json.JSONDecoder()
        parsed: Any = None
        ignored = ""
        for match in re.finditer(r"\[", text):
            try:
                parsed, consumed = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                ignored = text[match.start() + consumed:].strip()
                break
            parsed = None
        if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
            if "[" not in text and text.strip():
                return {"tool_name": "finish", "parameters": {"answer": text.strip()}}, ""
            raise ValueError("model did not emit one AgentTune Cohere-fallback JSON action")
        action = parsed[0]
        if set(action) != {"tool_name", "parameters"}:
            raise ValueError("action must contain exactly tool_name and parameters")
        if not isinstance(action["tool_name"], str) or not isinstance(action["parameters"], dict):
            raise ValueError("tool_name must be text and parameters must be an object")
        return action, ignored


def _event_json(log: Any) -> list[dict[str, Any]]:
    return [
        {
            "kind": str(getattr(event.kind, "value", event.kind)),
            "payload": event.payload,
        }
        for event in log
    ]


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 requires a Hugging Face GPU Job; CUDA is unavailable")

    from agenttune.agentic import DictToolHarness, Project, ReActStrategy
    from evals.benchmark.loader import load_benchmark
    from tools import Toolbox
    from tools.definitions import get_tool_schemas, validate_tool_arguments

    schemas = get_tool_schemas()
    verify_frozen_config(schemas)
    runtime_versions = verify_runtime_versions(torch)
    tasks = {task.task_id: task for task in load_benchmark("train-support")}
    task = tasks[TASK_ID]
    print(json.dumps({
        "event": "job_environment",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "task_id": TASK_ID,
        "device": torch.cuda.get_device_name(0),
        "runtime_versions": runtime_versions,
        "cuda": torch.version.cuda,
    }, sort_keys=True))

    started = time.monotonic()
    model_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=EVAL_CONFIG["model"]["trust_remote_code"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=getattr(torch, EVAL_CONFIG["model"]["dtype"]),
        device_map=EVAL_CONFIG["model"]["device_map"],
        trust_remote_code=EVAL_CONFIG["model"]["trust_remote_code"],
    )
    model.eval()
    model_load_seconds = time.monotonic() - model_started
    plain_messages = [
        {"role": "user", "content": "In one short sentence, what does a grep tool do?"}
    ]
    plain_inputs = tokenizer.apply_chat_template(
        plain_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    plain_prompt_length = plain_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        plain_generated = model.generate(
            **plain_inputs,
            max_new_tokens=EVAL_CONFIG["generation"]["plain_text_max_new_tokens"],
            do_sample=EVAL_CONFIG["generation"]["do_sample"],
            use_cache=EVAL_CONFIG["generation"]["use_cache"],
        )
    plain_text = tokenizer.decode(
        plain_generated[0, plain_prompt_length:], skip_special_tokens=True
    ).strip()
    print(json.dumps({"event": "plain_text_inference", "output": plain_text}))
    if not plain_text:
        raise RuntimeError("Tiny Aya returned an empty plain-text response")

    with tempfile.TemporaryDirectory(prefix="terminal-slm-hf-smoke-") as temp_root:
        workspace = Path(temp_root) / "workspace"
        copytree(task.snapshot_path, workspace)
        with Toolbox(workspace) as toolbox:
            tool_executions: list[dict[str, Any]] = []
            dispatch = {}
            for schema in schemas:
                name = schema["function"]["name"]

                def make_tool(tool_name: str):
                    def execute(**arguments: Any) -> dict[str, Any]:
                        validated = validate_tool_arguments(tool_name, arguments)
                        result = getattr(toolbox, tool_name)(**validated)
                        result_dict = result.as_dict()
                        tool_executions.append({
                            "tool": tool_name,
                            "arguments": validated,
                            "result": result_dict,
                        })
                        return result_dict
                    return execute

                dispatch[name] = make_tool(name)

            harness = DictToolHarness(dispatch, max_steps=MAX_STEPS, done_tool="finish")
            policy = TinyAyaPolicy(
                tokenizer,
                model,
                torch,
                schemas,
                time_limit_seconds=EVAL_CONFIG["agent"]["max_execution_seconds_per_task"],
            )
            strategy = ReActStrategy(policy, max_steps=MAX_STEPS)
            project = Project(strategy=strategy, harness=harness)

            failure: str | None = None
            event_log = None
            try:
                event_log = project.infer(task.instruction)
            except Exception as exc:  # record real smoke failures in the job logs
                failure = f"{type(exc).__name__}: {exc}"

            events = _event_json(event_log) if event_log is not None else []
            actions = [
                event["payload"].get("action", {})
                for event in events
                if "tool_call" in str(event["kind"]).lower()
            ]
            selected_tools = [
                action.get("name") for action in actions
                if action.get("name") and action.get("name") != "finish"
            ]
            finish_actions = [action for action in actions if action.get("name") == "finish"]
            final_answer = ""
            if finish_actions:
                final_answer = str(finish_actions[-1].get("arguments", {}).get("answer", ""))
            response_ok = all(
                term.casefold() in final_answer.casefold()
                for term in task.success_condition.get("response_contains", [])
            )
            required_purpose_terms = PURPOSE_TERMS if task.task_id == "B006" else ()
            purpose_ok = (
                all(term in final_answer.casefold() for term in required_purpose_terms)
                if required_purpose_terms
                else None
            )
            tools_ok = set(task.success_condition.get("required_tools", [])).issubset(selected_tools)
            validation_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "HOME": str(workspace),
                "TMPDIR": str(workspace / ".tmp"),
            }
            Path(validation_env["TMPDIR"]).mkdir(exist_ok=True)
            validation = subprocess.run(
                task.validation_command,
                cwd=workspace,
                env=validation_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=EVAL_CONFIG["agent"]["validation_timeout_seconds"],
                check=False,
            )
            validation_ok = validation.returncode == 0
            grep_succeeded = any(
                "tool_result" in str(event["kind"]).lower()
                and isinstance(event["payload"].get("output"), dict)
                and event["payload"]["output"].get("success") is True
                for event in events
            )
            success = (
                failure is None
                and response_ok
                and purpose_ok is not False
                and tools_ok
                and grep_succeeded
                and validation_ok
            )
            report = {
                "run_kind": "phase1_hf_gpu_smoke",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "config_id": EVAL_CONFIG["config_id"],
                "config_sha256": _canonical_sha256(EVAL_CONFIG),
                "model_load_seconds": round(model_load_seconds, 3),
                "task_id": TASK_ID,
                "split": task.split,
                "success": success,
                "failure": failure,
                "selected_tools": selected_tools,
                "final_answer": final_answer,
                "final_response_source": policy.final_response_source,
                "plain_text_output": plain_text,
                "response_condition_passed": response_ok,
                "purpose_condition_passed": purpose_ok,
                "required_purpose_terms": required_purpose_terms,
                "required_tools_passed": tools_ok,
                "tool_observation_passed": grep_succeeded,
                "validation_exit_code": validation.returncode,
                "validation_stdout": validation.stdout,
                "validation_stderr": validation.stderr,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "model_outputs": policy.raw_outputs,
                "model_prompts": policy.prompts,
                "model_parse_errors": policy.parse_errors,
                "selected_actions": policy.selected_actions,
                "returned_action_output_indices": policy.returned_action_output_indices,
                "prompt_tokens": policy.prompt_tokens_seen,
                "generated_tokens": policy.generated_tokens_seen,
                "ignored_trailing_model_text": policy.ignored_trailing_text,
                "response_adaptations": policy.response_adaptations,
                "tool_executions": tool_executions,
                "trajectory": events,
            }
            print("PHASE1_SMOKE_RESULT=" + json.dumps(report, ensure_ascii=False, default=str))
            return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
