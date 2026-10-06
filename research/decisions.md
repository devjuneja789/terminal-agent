# Research decisions

This file records project decisions and their rationale. Phase A1 establishes the initial tool boundary; model and benchmark decisions remain open.

## Initial direction

- **Starting model:** `Qwen/Qwen3-1.7B`.
- **Target behavior:** reliable use of a constrained terminal/tool interface for executable coding tasks.
- **Evaluation:** compare the unmodified model and adapted checkpoints using an executable benchmark with objective success checks.
- **Training:** QLoRA SFT is the planned first training method; DPO and GRPO remain optional and depend on evidence, data, and compute.
- **Compute split:** local machine for development, evaluation, and quantized inference; cloud GPU for training.
- **Current status:** Phase A1 (foundational tool layer) implemented and covered by unit tests. A2 has not started; no inference, benchmark, or training work has been done.

## Phase A1 design decisions

- **One explicit workspace per toolbox:** file tools accept only workspace-relative paths and verify resolved paths remain inside the workspace, including symlinks. The home directory itself and any ancestor of it cannot be selected as the workspace root.
- **Temporary default:** omitting a workspace creates a new temporary directory. It does not default to the process cwd or the user's home directory.
- **Typed result envelope:** all tools return `ToolResult` with success, stdout, stderr, exit status, error, truncation flags, and metadata; `as_dict()` provides a serialization boundary.
- **Bounded subprocesses:** shell and Git use a 30-second default timeout (maximum 300 seconds), separate 64 KiB output limits, no inherited stdin, and process-group termination on timeout.
- **Filtered child environment:** only selected locale/terminal/path values pass through; home, temp, XDG, and Git config locations resolve within the workspace. Git discovery is prevented from walking above the workspace parent, network subcommands are denied, and only a small built-in subcommand set is exposed.
- **Command audit:** shell and Git invocations are recorded as JSONL in `.terminal-slm/tool.log.jsonl` inside the workspace.
- **No extra runtime dependencies:** the Phase A1 tool layer uses Python's standard library and is exercised with `unittest`.
- **Safety limit:** the shell has a pinned cwd and conservative command checks, but it is not a kernel/container sandbox. The future agent loop must enforce the workspace policy and avoid treating untrusted model output as safe.

## Decisions to resolve before A2 and later phases

- Define how the agent loop represents tool calls, observations, errors, and conversation state.
- Choose the initial model client/local runtime interface and confirm model artifact access before inference.
- Decide whether A2 may execute arbitrary shell commands under the documented guardrails or needs stronger OS-level isolation.
- Define tool approval/denial behavior, token/turn limits, and interruption handling.
- Define benchmark task format, success criteria, and result-recording format before generating data or reporting metrics.
- Pin the development/inference environment after checking compatibility on the actual target machines; keep training dependencies separate and target a verified cloud GPU.
