# Terminal-SLM

Terminal-SLM is a research project to build a small language model specialized for reliable terminal and tool use, embed it in a local terminal agent, and evaluate it on executable coding tasks.

## Research question

Can supervised fine-tuning, with optional preference or reward optimization, make Qwen3-1.7B more reliable at using a constrained terminal toolset to complete executable coding tasks than the unmodified model, under local inference constraints?

## Planned phases

- **A — Agent harness:** connect the selected model to a constrained terminal/tool interface.
- **B — Executable benchmark:** define reproducible coding tasks, isolated workspaces, and objective success checks.
- **C — Baseline evaluation:** evaluate the unmodified model and record reproducible metrics before training.
- **D — QLoRA SFT:** train on curated terminal/tool-use demonstrations using cloud GPU resources.
- **E — Optional DPO:** test preference optimization if suitable preference data and a clear need exist.
- **F — Optional GRPO:** test reward-based optimization if the benchmark reward is stable and cloud compute is available.
- **G — GGUF and local deployment:** quantize the selected checkpoint and run it with llama.cpp locally; compare it against the baseline.

The intended order is **Qwen3-1.7B → tool-calling agent → executable benchmark → baseline evaluation → QLoRA SFT → optional DPO → optional GRPO → GGUF → local llama.cpp deployment**. No phase has started in this repository; this checkout is currently for reconnaissance and planning only.

## Hardware constraints

The project context specifies a local NVIDIA GTX 1060 Max-Q with 6 GB VRAM. The current WSL2 environment cannot see an NVIDIA device or driver interface, so this hardware detail has not been verified from inside this environment. Local use is intended for development, evaluation, and quantized inference. Training is expected to use cloud GPUs; no cloud account, GPU, or credentials have been checked or assumed.

The current environment has about 7.7 GiB RAM and 283 GiB free on the workspace filesystem. It has Python 3.12.3, but no PyTorch or model-training packages are installed. See [DEVELOPMENT.md](DEVELOPMENT.md) for inspection details and limits.

## What is currently available

- WSL2 on Ubuntu 24.04.1 LTS, x86_64, with Python 3.12.3 and pip 24.0.
- Git global identity and GitHub credential-helper configuration; the workspace itself is not a functioning Git repository.
- The existing `implementation_plan_claude.md` planning note.
- Public Hugging Face listing for [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). This confirms the repository is listed publicly, but model weights have not been downloaded and download access from this machine has not been tested.

## What is missing or unverified

- A usable Git repository/commit history in this workspace.
- PyTorch, Transformers, Datasets, TRL, PEFT, bitsandbytes, Unsloth, Hugging Face Hub client, and llama.cpp.
- Accessible NVIDIA driver/CUDA device in the current WSL2 environment; `nvidia-smi` reports GPU access blocked and `nvcc` is absent.
- A local Qwen3-1.7B model cache, verified model download, and verified inference path.
- Agent implementation, benchmark tasks, baseline results, training data, trained checkpoints, and deployment artifacts.
- Any verified cloud GPU access, account, credentials, or allocation.

No dependencies were installed and no benchmark or model result has been generated during reconnaissance.
