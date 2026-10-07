# Research decisions

This file records project decisions and their rationale. Phases A1–A3 establish the tool boundary, native model/tool-call interface, and bounded execution loop; benchmark and training decisions remain open.

## Initial direction

- **Starting model:** `Qwen/Qwen3-1.7B`.
- **Target behavior:** reliable use of a constrained terminal/tool interface for executable coding tasks.
- **Evaluation:** compare the unmodified model and adapted checkpoints using an executable benchmark with objective success checks.
- **Training:** QLoRA SFT is the planned first training method; DPO and GRPO remain optional and depend on evidence, data, and compute.
- **Compute split:** local machine for development, evaluation, and quantized inference; cloud GPU for training.
- **Current status:** A1–A3 are implemented and covered by unit tests. Real model inference, benchmark, and training have not been run.

## Phase A1 design decisions

- **One explicit workspace per toolbox:** file tools accept only workspace-relative paths and verify resolved paths remain inside the workspace, including symlinks. The home directory itself and any ancestor of it cannot be selected as the workspace root.
- **Temporary default:** omitting a workspace creates a new temporary directory. It does not default to the process cwd or the user's home directory.
- **Typed result envelope:** all tools return `ToolResult` with success, stdout, stderr, exit status, error, truncation flags, and metadata; `as_dict()` provides a serialization boundary.
- **Bounded subprocesses:** shell and Git use a 30-second default timeout (maximum 300 seconds), separate 64 KiB output limits, no inherited stdin, and process-group termination on timeout.
- **Filtered child environment:** only selected locale/terminal/path values pass through; home, temp, XDG, and Git config locations resolve within the workspace. Git discovery is prevented from walking above the workspace parent, network subcommands are denied, and only a small built-in subcommand set is exposed.
- **Command audit:** shell and Git invocations are recorded as JSONL in `.terminal-slm/tool.log.jsonl` inside the workspace.
- **No extra runtime dependencies:** the Phase A1 tool layer uses Python's standard library and is exercised with `unittest`.
- **Safety limit:** the shell has a pinned cwd and conservative command checks, but it is not a kernel/container sandbox. The future agent loop must enforce the workspace policy and avoid treating untrusted model output as safe.

## Phase A2 model and protocol decisions

- **Keep model inference separate from tool orchestration:** `ModelClient` accepts conversation messages and standard tool schemas and returns raw assistant text. The model client does not validate or execute tool calls.
- **Use Qwen3's native tool-call format:** parse `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` directly rather than introducing a project-specific syntax. This keeps the base model, future SFT traces, and later local deployment aligned with the format in the project plan.
- **Use standard OpenAI-style tool schemas as model input:** the five existing workspace tools have JSON parameter schemas in `tools/definitions.py`, separate from inference and execution. This is the schema interface expected by the tokenizer chat template and other compatible runtimes.
- **Validate before execution:** the parser validates all blocks in an assistant response (JSON shape, known tool, required and allowed fields, argument types and ranges) before returning calls. The executor repeats validation at the execution boundary. An invalid later block therefore cannot partially execute an earlier call from the same response.
- **Sequential loop:** each model turn may return one or more ordered tool calls. The loop executes them in order, appends native `tool` role results, and then asks the model for the next turn. A response with no tool-call block ends the interaction as normal assistant text.
- **Start with local Transformers, without implicit downloads:** `LocalTransformersClient` uses the Qwen3 tokenizer's own chat template with the tool schemas and returns raw decoded text. Imports/model loading are lazy, and `local_files_only=True` is the default. A local model runtime and checkpoint are not present in the inspected environment, so real inference remains unverified.
- **No remote client yet:** a remote OpenAI-compatible adapter is unnecessary to test the interface; tests use a fake in-process `ModelClient`.
- **No training or benchmark work in A2:** tests verify parsing/orchestration behavior only and are not model evaluation results.

## Phase A3 loop and state decisions

- **Keep the loop deterministic:** one model response is parsed, all calls in that response are validated, and calls execute in emitted order. Results are appended as tool messages before the next inference.
- **Bound run resources:** `AgentConfig` caps tool calls, elapsed time, total conversation characters, and each serialized tool observation. The model and shell/Git tools receive the remaining time budget where supported; after a deadline, the loop starts no further action.
- **Return explicit stop states:** completion, tool-call limit, execution timeout, conversation limit, malformed tool call, and model failure are represented distinctly. Tool failures become observations so the model can recover; parser validation failures stop before execution.
- **Record a timestamped trajectory:** the trace stores the user request, each raw model response, each tool name and arguments, bounded tool results, timestamps, final response, and stop reason. Truncation is marked in the event/result.
- **No planning or persistent memory:** conversation state is limited to the current task and is discarded with the run result unless a caller stores its trajectory.
- **Timeout limitation:** Python operations such as file reads and grep cannot be forcibly interrupted safely. The loop checks the deadline between operations; subprocess tools and model clients are expected to honor the remaining-time contract.

## Decisions to resolve before later phases

- Decide whether the current shell guardrails are sufficient or need stronger OS-level isolation.
- Define tool approval/denial behavior, token/turn limits, and interruption handling.
- Install and pin a local inference runtime, confirm model artifact access, and run a real smoke inference before baseline evaluation.
- Define benchmark task format, success criteria, and result-recording format before generating data or reporting metrics.
- Pin the development/inference environment after checking compatibility on the actual target machines; keep training dependencies separate and target a verified cloud GPU.
