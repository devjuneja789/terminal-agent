"""One real Tiny Aya + AgentTune tool-use smoke task for HF GPU Jobs.

This file is staged with only the Terminal-SLM tool layer and B006's
train-support snapshot. It deliberately does not run the formal evaluator or
touch any held-out task.
"""

from __future__ import annotations

import importlib.metadata
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

MODEL_ID = "CohereLabs/tiny-aya-global"
TASK_ID = "B002"  # train-support only: locate the route table using grep
MAX_STEPS = 6
MAX_NEW_TOKENS = 256
PURPOSE_TERMS = ("maps", "method", "path", "route")


class TinyAyaPolicy:
    """Tiny Aya policy using AgentTune's documented Cohere fallback format.

    The format is a JSON list of {tool_name, parameters} objects. Tool
    observations are supplied as user turns, matching AgentTune's Cohere
    fallback because Tiny Aya's chat template does not render tool-role turns.
    """

    def __init__(self, tokenizer: Any, model: Any, torch: Any, schemas: list[dict[str, Any]]) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch
        self.schemas = schemas
        self.raw_outputs: list[str] = []
        self.assistant_turns: list[str] = []
        self.ignored_trailing_text: list[str] = []
        self.response_adaptations: list[str | None] = []
        self.prompts: list[list[dict[str, str]]] = []
        self.parse_errors: list[str | None] = []

    def __call__(self, state: Any) -> dict[str, Any]:
        messages = self._messages(state)
        self.prompts.append(messages)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        prompt_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        raw = self.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        self.raw_outputs.append(raw)
        try:
            action, trailing = self._parse_action(raw)
        except Exception as exc:
            self.parse_errors.append(f"{type(exc).__name__}: {exc}")
            raise
        self.parse_errors.append(None)
        adaptation: str | None = None
        # Tiny Aya sometimes emits one valid fallback action followed by its
        # ordinary assistant answer instead of a `finish` action. After at
        # least one observed tool step, treat a plain text tail as the final
        # response. Never execute the later proposed action from that same
        # completion. On the initial turn, keep the selected action so the
        # requested tool still runs.
        if getattr(state, "step", 0) > 0 and trailing and "[" not in trailing:
            discarded_action = action["tool_name"]
            action = {"tool_name": "finish", "parameters": {"answer": trailing}}
            adaptation = f"plain_text_tail_to_finish; discarded_action={discarded_action}"
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
        system = (
            "You are a terminal agent restricted to the supplied workspace. Choose the one "
            "most relevant tool for the user's task. For source-code searches, prefer grep; "
            "use shell or git only when the task needs a command or Git information. Paths are "
            "workspace-relative; use '.' to search the whole workspace. Call one tool at a time. "
            "When asked for a function's purpose, a signature alone is never enough: after "
            "grep finds it, call read_file on the matching source lines to inspect the body. "
            "Do not finish until you have seen the implementation. After each result, choose "
            "the next needed tool, then finish with a concise factual answer.\n\n"
            "Tools and required argument names:\n" + "\n".join(visible_tools) +
            "\n- finish(answer: string): Return the final response.\n\n"
            "Return exactly one JSON array with one action and no other text. Examples:\n"
            '[{"tool_name":"grep","parameters":{"pattern":"def example_function","path":"."}}]\n'
            '[{"tool_name":"read_file","parameters":{"path":"package/module.py","start_line":10,"end_line":20}}]\n'
            '[{"tool_name":"finish","parameters":{"answer":"Found it in package/module.py."}}]'
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        observations = []
        for event in state.events:
            kind = str(getattr(event.kind, "value", event.kind)).lower()
            payload = event.payload if isinstance(event.payload, dict) else {}
            if "observation" in kind and payload.get("text"):
                observations.append(str(payload["text"]))
        if observations:
            messages.append({"role": "user", "content": observations[0]})
            for index, observation in enumerate(observations[1:]):
                if index < len(self.raw_outputs):
                    messages.append({"role": "assistant", "content": self.assistant_turns[index]})
                messages.append({"role": "user", "content": "Tool result:\n" + observation})
        return messages

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

    tasks = {task.task_id: task for task in load_benchmark("train-support")}
    task = tasks[TASK_ID]
    print(json.dumps({
        "event": "job_environment",
        "model_id": MODEL_ID,
        "task_id": TASK_ID,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "agenttune": importlib.metadata.version("agenttune"),
        "cuda": torch.version.cuda,
    }, sort_keys=True))

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
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
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
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
            schemas = get_tool_schemas()
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
            policy = TinyAyaPolicy(tokenizer, model, torch, schemas)
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
                timeout=120,
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
                "task_id": TASK_ID,
                "split": task.split,
                "success": success,
                "failure": failure,
                "selected_tools": selected_tools,
                "final_answer": final_answer,
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
                "ignored_trailing_model_text": policy.ignored_trailing_text,
                "response_adaptations": policy.response_adaptations,
                "tool_executions": tool_executions,
                "trajectory": events,
            }
            print("PHASE1_SMOKE_RESULT=" + json.dumps(report, ensure_ascii=False, default=str))
            return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
