# Terminal-SLM

Terminal-SLM is a research project to build a small language model specialized for reliable terminal and tool use, embed it in a local terminal agent, and evaluate it on executable coding tasks.

## Research question

Can supervised fine-tuning, with optional preference or reward optimization, make Qwen3-1.7B more reliable at using a constrained terminal toolset to complete executable coding tasks than the unmodified model, under local inference constraints?

## Planned phases

- **A — Agent harness:** connect the selected model to a constrained terminal/tool interface. **A1 (tool layer), A2 (model interface/native tool-call parsing), A3 (bounded loop/state/trajectory), and A4 (CLI) are implemented.**
- **B — Executable benchmark:** define reproducible coding tasks, isolated workspaces, and objective success checks.
- **C — Baseline evaluation:** evaluate the unmodified model and record reproducible metrics before training.
- **D — QLoRA SFT:** train on curated terminal/tool-use demonstrations using cloud GPU resources.
- **E — Optional DPO:** test preference optimization if suitable preference data and a clear need exist.
- **F — Optional GRPO:** test reward-based optimization if the benchmark reward is stable and cloud compute is available.
- **G — GGUF and local deployment:** quantize the selected checkpoint and run it with llama.cpp locally; compare it against the baseline.

The intended order is **Qwen3-1.7B → tool-calling agent → executable benchmark → baseline evaluation → QLoRA SFT → optional DPO → optional GRPO → GGUF → local llama.cpp deployment**. Phases A1–A4 are implemented. Actual model inference, benchmark work, and training have not run.

## Phase A1 architecture

`tools.Toolbox` binds the five agent-facing operations to one `Workspace`:

- `shell(command, timeout)` runs Bash from the workspace root with a 30-second default timeout, a 300-second maximum, 64 KiB stdout/stderr caps, and JSONL command logging.
- `read_file(path, start_line, end_line)` and `write_file(path, content)` use workspace-relative paths and reject absolute paths, traversal, and symlink escapes.
- `grep(pattern, path)` searches one file or recursively searches a directory within the workspace, returning up to 1,000 matches.
- `git(command)` runs a restricted set of Git subcommands from the workspace, without shell interpolation or network operations.

If no workspace is provided, `Toolbox()` creates a temporary directory outside the user's home directory. A caller can pass an explicit task/project directory or use the API as a context manager:

```python
from tools import Toolbox

with Toolbox() as tools:
    result = tools.shell("pwd")
```

Subprocesses receive an allowlisted environment; `HOME`, temporary paths, XDG paths, and Git configuration are redirected into the workspace. Shell commands are logged at `.terminal-slm/tool.log.jsonl`. Obvious destructive operations (such as recursive `rm`, `sudo`, `find -delete`, forced Git cleanup, and history rewriting) are denied.

The tool layer is a workspace-scoped guardrail, not an operating-system sandbox: shell commands execute arbitrary programs, and a deliberately supplied absolute path can still be accessed by those programs. The agent loop must preserve the workspace boundary and treat command execution as a privileged capability.

## Phase A2 model and tool-call protocol

The inference boundary is `ModelClient.generate(messages, tools, timeout=None) -> str`: it returns raw assistant text and does not parse calls or execute tools. `LocalTransformersClient` is the initial local implementation. It supplies the OpenAI-style tool schemas to Qwen3's own tokenizer chat template, then returns decoded output without rewriting the model's native protocol. Model loading is lazy and `local_files_only=True` by default, so inference does not silently download weights.

The protocol remains Qwen3's native format:

```text
<tool_call>{"name":"write_file","arguments":{"path":"notes.txt","content":"hello"}}</tool_call>
```

Tool definitions live separately in `tools/definitions.py`. `agent/parser.py` parses each native block and validates JSON, tool name, required fields, types, ranges, and unexpected fields. It validates every call in an assistant response before the executor can run any of them. `agent/loop.py` handles sequential model/tool turns; `tools/executor.py` dispatches validated calls to the existing workspace-bound `Toolbox`. Plain assistant text with no tool-call block is treated as the final response. There is no remote-host client or alternate call syntax.

The local Transformers inference adapter is implemented but not runnable in the inspected environment yet: PyTorch, Transformers, and local Qwen3 model files are absent. A loopback-only OpenAI-compatible client can connect to a local inference server; it does not use credentials or send prompts to remote hosts. Install/use model dependencies only when preparing to run real inference. Tests use fake `ModelClient` instances and do not fabricate model outputs as evaluation results.

The implementation uses the Python standard library for parsing, schemas, orchestration, CLI, and tests. Its suite runs with `python3 -m unittest discover -v`; actual `LocalTransformersClient` generation requires the optional `torch` and `transformers` packages.

## Phase A3 bounded agent loop

`AgentLoop` receives a user request, passes the five schemas to `ModelClient`, validates each response before execution, appends each tool observation as a `tool` message, and repeats until normal assistant text or a stop condition. `AgentConfig` sets maximum tool calls, wall-clock execution time, conversation characters, and serialized tool-output characters. Calls beyond the tool limit are recorded as skipped; oversized observations are clipped and marked. The loop records timestamped model/tool events, request, arguments, results, final response, and stop reason in `AgentTrajectory.to_dict()`.

The deadline is cooperative across Python operations: it is passed to the model client and to shell/Git timeout handling, and the loop will not start another action after the deadline. A model backend must honor its timeout contract; synchronous file/grep operations cannot be forcibly interrupted mid-call. The loop does not add planning or long-term memory.

## Phase A4 CLI

The source checkout includes a `term-agent` launcher and `cli.py`. Run a task against an explicit project directory with:

```sh
./term-agent --workspace ./my-project "Fix the failing authentication tests"
```

The launcher is also named `term-agent`; add the checkout directory to `PATH` to invoke it without the `./` prefix.

Without `--workspace`, the CLI creates a fresh temporary directory outside the home directory and removes it at exit. This default is isolated and starts empty; pass a project/task directory explicitly when the agent needs to inspect existing files. The tool layer still enforces workspace-relative file operations and its shell guardrails; it is a workspace-scoped guardrail, not an operating-system sandbox.

Tasks can also be piped from scripts or entered interactively when no task argument is given:

```sh
printf '%s\n' 'Summarize the workspace files' | ./term-agent --workspace ./my-project
```

Use `--model-path /path/to/checkpoint` for local Transformers inference (default: `Qwen/Qwen3-1.7B`, local files only), or `--model-endpoint http://127.0.0.1:8080/v1/chat/completions --model-name Qwen3-1.7B` for a local OpenAI-compatible server. Endpoints are restricted to loopback addresses. `--max-steps N` sets the tool-call limit. The CLI prints model/action summaries, tool results, and the final response. Exit status is `0` on completion, `2` for input/workspace errors, `3` for the step limit, `4` for time/conversation limits, and nonzero for model or agent errors.

The path through the harness is:

```text
task input → CLI/workspace → ModelClient + five tool schemas → AgentLoop
                                                    ↑            ↓
                                         model response ← ToolExecutor/Toolbox
```

The CLI logs the completed trajectory in order and displays bounded tool observations; detailed timestamps, arguments, outcomes, and stop reason remain available in `AgentResult.trajectory` for future integrations.

## Hardware constraints

The local NVIDIA GTX 1060 Max-Q with 6 GB VRAM has been confirmed from a plain WSL terminal. The recorded host check showed driver 582.78 and CUDA driver support through 13.0; the sandboxed inspection used for this project did not expose the GPU, and `nvcc`/CUDA toolkit availability remains unverified. Local use is intended for development, evaluation, and quantized inference. Training is expected to use cloud GPUs; no cloud account, GPU, or credentials have been checked or assumed.

The current environment has about 7.7 GiB RAM and 283 GiB free on the workspace filesystem. It has Python 3.12.3, but no PyTorch or model-training packages are installed. See [DEVELOPMENT.md](DEVELOPMENT.md) for inspection details and limits.

## What is currently available

- WSL2 on Ubuntu 24.04.1 LTS, x86_64, with Python 3.12.3 and pip 24.0.
- A Git repository on `main` tracking `origin/main`, with global identity and GitHub credential-helper configuration.
- The Phase A1 `Toolbox` implementation and its standard-library unit tests.
- The A2 native tool-call client/parser and A3 bounded loop with scripted mock-client tests.
- The existing `implementation_plan_claude.md` planning note.
- Public Hugging Face listing for [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). This confirms the repository is listed publicly, but model weights have not been downloaded and download access from this machine has not been tested.

## What is missing or unverified

- PyTorch, Transformers, Datasets, TRL, PEFT, bitsandbytes, Unsloth, Hugging Face Hub client, and llama.cpp.
- PyTorch and a verified PyTorch CUDA runtime; CUDA toolkit/compiler availability also remains unverified.
- A local Qwen3-1.7B model cache, verified model download, and verified inference path.
- A runnable local inference environment and cached Qwen3-1.7B checkpoint, benchmark tasks, baseline results, training data, trained checkpoints, and deployment artifacts.
- Any verified cloud GPU access, account, credentials, or allocation.

No model dependencies were installed, and no benchmark or model result has been generated. Parser and agent-loop unit tests use scripted mock responses, not model results.
