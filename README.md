# Terminal-SLM

Terminal-SLM is a research project to build a small language model specialized for reliable terminal and tool use, embed it in a local terminal agent, and evaluate it on executable coding tasks.

## Research question

Can supervised fine-tuning, with optional preference or reward optimization, make Qwen3-1.7B more reliable at using a constrained terminal toolset to complete executable coding tasks than the unmodified model, under local inference constraints?

## Planned phases

- **A — Agent harness:** connect the selected model to a constrained terminal/tool interface. **A1 (tool layer) is implemented; A2 (agent loop/model integration) has not started.**
- **B — Executable benchmark:** define reproducible coding tasks, isolated workspaces, and objective success checks.
- **C — Baseline evaluation:** evaluate the unmodified model and record reproducible metrics before training.
- **D — QLoRA SFT:** train on curated terminal/tool-use demonstrations using cloud GPU resources.
- **E — Optional DPO:** test preference optimization if suitable preference data and a clear need exist.
- **F — Optional GRPO:** test reward-based optimization if the benchmark reward is stable and cloud compute is available.
- **G — GGUF and local deployment:** quantize the selected checkpoint and run it with llama.cpp locally; compare it against the baseline.

The intended order is **Qwen3-1.7B → tool-calling agent → executable benchmark → baseline evaluation → QLoRA SFT → optional DPO → optional GRPO → GGUF → local llama.cpp deployment**. Phase A1 is the first implemented subphase; no model inference, benchmark work, or training has started.

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

The implementation uses only the Python standard library. Its unit tests run with `python3 -m unittest discover -v`.

## Hardware constraints

The local NVIDIA GTX 1060 Max-Q with 6 GB VRAM has been confirmed from a plain WSL terminal. The recorded host check showed driver 582.78 and CUDA driver support through 13.0; the sandboxed inspection used for this project did not expose the GPU, and `nvcc`/CUDA toolkit availability remains unverified. Local use is intended for development, evaluation, and quantized inference. Training is expected to use cloud GPUs; no cloud account, GPU, or credentials have been checked or assumed.

The current environment has about 7.7 GiB RAM and 283 GiB free on the workspace filesystem. It has Python 3.12.3, but no PyTorch or model-training packages are installed. See [DEVELOPMENT.md](DEVELOPMENT.md) for inspection details and limits.

## What is currently available

- WSL2 on Ubuntu 24.04.1 LTS, x86_64, with Python 3.12.3 and pip 24.0.
- A Git repository on `main` tracking `origin/main`, with global identity and GitHub credential-helper configuration.
- The Phase A1 `Toolbox` implementation and its standard-library unit tests.
- The existing `implementation_plan_claude.md` planning note.
- Public Hugging Face listing for [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). This confirms the repository is listed publicly, but model weights have not been downloaded and download access from this machine has not been tested.

## What is missing or unverified

- PyTorch, Transformers, Datasets, TRL, PEFT, bitsandbytes, Unsloth, Hugging Face Hub client, and llama.cpp.
- PyTorch and a verified PyTorch CUDA runtime; CUDA toolkit/compiler availability also remains unverified.
- A local Qwen3-1.7B model cache, verified model download, and verified inference path.
- A2 agent loop, model interface/inference, benchmark tasks, baseline results, training data, trained checkpoints, and deployment artifacts.
- Any verified cloud GPU access, account, credentials, or allocation.

No model dependencies were installed, and no benchmark or model result has been generated. Phase A1 unit tests use only the Python standard library.
