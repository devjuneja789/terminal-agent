# Research decisions

This file records project decisions and their rationale. Phases A1–A4 establish the tool boundary, native model/tool-call interface, bounded execution loop, and CLI. The manual Qwen3 smoke gate is prepared but blocked on local inference prerequisites; benchmark and training decisions remain open.

> Historical note: the initial paragraphs below describe the original Qwen3 plan. The active hackathon pivot and current phase status are recorded in `HACKATHON-PIVOT-TRACK-3.md` and EXP-003.

## Initial direction

- **Starting model:** `Qwen/Qwen3-1.7B`.
- **Target behavior:** reliable use of a constrained terminal/tool interface for executable coding tasks.
- **Evaluation:** compare the unmodified model and adapted checkpoints using an executable benchmark with objective success checks.
- **Training:** QLoRA SFT is the planned first training method; DPO and GRPO remain optional and depend on evidence, data, and compute.
- **Compute split:** local machine for development, evaluation, and quantized inference; cloud GPU for training.
- **Current status:** A1–A4 are implemented and covered by unit tests. The five-task manual smoke runner is implemented. A local checkpoint loaded on 2026-10-16, but generation failed because the client treated Transformers `BatchEncoding` as a tensor. The client now passes the encoded mapping to `generate`; the manual smoke gate still needs a successful rerun. The formal benchmark and training remain incomplete.

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

## Phase B initial benchmark decisions

- **Versioned, pinned repository snapshot:** benchmark tasks use the compact Python miniapp under `evals/benchmark/snapshots/miniapp`. Every record includes a content digest; loading rejects a changed snapshot. Each task starts from a separate workspace copy so one edit cannot affect another task or the source snapshot.
- **Fixed stratified split:** version `terminal-slm-initial-v1` contains 50 tasks, five in each of ten categories. Four tasks per category (40 total) are train-support; one per category (10 total) is held-out evaluation. The permanent IDs are in `evals/benchmark/manifest.json`.
- **Contamination rule:** held-out tasks and any future outcomes/trajectories derived from them must not be used to make SFT, DPO, or GRPO examples, preferences, demonstrations, or reward data. `training_task_ids()` exposes train-support IDs only. This is a data-handling policy as repository files themselves are not access controlled.
- **Executable task contract:** task records include the instruction, expected outcome, validation command, and success conditions. The runner checks final-answer terms and required tool names as applicable, then executes validation in the isolated task workspace. Optional reference patches are omitted from this initial set.
- **Initial support-split diagnostic run:** actual Qwen3-1.7B inference was run on ten train-support tasks across two batches. Two tasks passed; tool-call schemas were valid for emitted calls, while wrong tool/path choices, one validation failure, and a timeout caused the other failures. This is diagnostic evidence only, not a full baseline. JSONL trajectories and per-run summaries are stored under `evals/benchmark/results/train-support/`. No held-out tasks were run and no training examples were created.
- **Scope limit:** the miniapp is useful for validating the initial executable format but does not establish broad real-repository coverage. Before baseline evaluation, review realism, overlap, difficulty, validator quality, and snapshot diversity; a replacement requires a new version and a newly frozen held-out set.
- **B2 evaluation records:** each run selects exactly one split, checks selected IDs against it before model loading, and writes complete trajectories, validation output, outcomes, token counts when the local runtime provides exact counters, and summary metrics. The evaluator never creates SFT/DPO/GRPO examples. Temporary task workspaces are removed after records are written.
- **Runtime constraint observed:** the evaluator session could not access the GTX 1060 and ran the local checkpoint on CPU. Cold loading took about 4.5 minutes and one task exceeded its 600-second execution budget because generation deadlines are cooperative. Use the WSL GPU-enabled runtime for practical larger baseline runs; retain the CPU traces as actual but limited diagnostic observations.

## Phase A manual Qwen3 smoke gate

- **Keep the first smoke tasks disposable and small:** the suite checks locating a function, finding an environment variable, reading a file, running one existing unit test, and making one marker-file change. Each task operates on a fresh temporary copy of the repository; this avoids allowing the trivial change task to modify the real checkout.
- **Persist complete run evidence:** `evals/manual_smoke.py` writes one JSONL record per task with the full timestamped `AgentTrajectory`, raw model outputs when available, tool arguments/results, model/tool step counts, latency, stop reason, and task verifier outcome. This suite is an integration diagnostic, not the formal Phase B benchmark.
- **Do not convert setup failures into model failures:** the 2026-10-10 smoke attempt ended on the first model turn for all five tasks. `torch`, `transformers`, and the Qwen3 cache/weights were absent; NVML/GPU access was blocked in this execution session. The recorded stop reason is `model_error`, with no tool calls. This diagnoses an unavailable model-inference environment. It gives no evidence about tool schemas, parser behavior, executor behavior, Qwen3 context formatting, or the model's task capability.
- **Current result artifact:** `research/results/manual_smoke_20261010_environment_blocked.jsonl` contains the five actual setup-failure trajectories. No synthetic model output or benchmark score was generated.
- **Gate remains open:** after a compatible runtime and locally available base weights are provided, rerun the same tasks, inspect each complete trace, and diagnose any failures by layer before claiming an end-to-end milestone or beginning Phase B.
- **2026-10-16 owner run:** the default CLI path selected `Qwen/Qwen3-1.7B` and failed to load offline because weights were stored at `models/Qwen3-1.7B`; use an explicit `--model-path`. With the explicit path, 311 model weight shards loaded, but generation failed at `rendered.shape`: this Transformers version returned a `BatchEncoding`, not a tensor. `LocalTransformersClient` now uses `return_dict=True`, reads `input_ids.shape`, and passes the mapping as keyword arguments to `generate`. A unit regression test simulates that object without loading packages or weights. Rerun the smoke tasks with the explicit local path to verify real inference; no successful end-to-end task is claimed yet.
- **2026-10-16 smoke follow-up:** the owner reran the suite and all five tasks completed with successful tool execution. The original output reported 3/5 because two verifiers were too strict: unittest wrote `OK` to stderr, and the requested marker was written without a trailing newline. The verifiers now account for both. The saved trajectories support 5/5 under corrected checks; a fresh run is recommended so the JSONL directly records that verifier revision.

## Hackathon Phase 2 decisions (2026-11-04)

- **Pin model and software before baseline:** `configs/tiny_aya_eval.json` pins the Tiny Aya Hub revision, FP16, Python 3.12, PyTorch 2.11.0, Transformers 5.22.1, Accelerate 1.15.0, AgentTune 1.1.3, prompt, and tool-schema hashes. This avoids silently changing the before/after evaluation environment.
- **Use the AgentTune Cohere path:** Tiny Aya uses AgentTune's Cohere fallback action format; the original Qwen native parser remains a separate historical route. AgentTune handles rollouts and tool orchestration, while Terminal-SLM's existing workspace tools and executable validators remain the authority for task state.
- **Freeze rather than edit benchmark:** the benchmark version and all ten held-out IDs are asserted before inference. No benchmark task or split has been modified in Phase 2.
- **Make held-out access explicit:** the cloud baseline bundle contains the held-out JSONL and a held-out-only manifest, but no support JSONL. Training rollout generation must use a separately prepared support-only bundle.
- **Separate setup from results:** Phase 2 creates the evaluator and launch path only. No model inference, baseline result, training run, or synthetic data was generated. Phase 3 is gated on the user's launch after checking live GPU Job cost and HF model access.
- **Metric definitions:** malformed model outputs count as invalid tool attempts. Tool-selection accuracy is aggregated across tasks with the benchmark's required-versus-unexpected distinct tool formula. Task success still requires response, required tool, validator, and completion checks.
