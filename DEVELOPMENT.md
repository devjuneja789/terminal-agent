# Development environment reconnaissance

Inspection date: 2026-10-02 (UTC)

## Environment assumptions and limits

This report originally described GPU access as unavailable from inside WSL. That finding has since been corrected: it was an artifact of running the inspection through a sandboxed Codex CLI session, not a real WSL/driver problem. Running `nvidia-smi` directly in a plain WSL terminal (host `DESKTOP-3UB5URJ`, working dir `/mnt/e/terminal-agent`, 2026-10-02 19:14 UTC) confirms the GPU is visible and working:

```
NVIDIA-SMI 580.178.01   Driver Version: 582.78   CUDA Version: 13.0
GPU 0: NVIDIA GeForce GTX 1060 (Max-Q) — 0MiB / 6144MiB used, 0% util, P8, 54C
```

Treat the rest of this report's figures (RAM, installed packages, etc.) that were captured through the same Codex-sandboxed session as unverified until re-checked directly in a real WSL shell — the GPU miss shows that sandbox doesn't reliably reflect the host environment.

The Qwen3-1.7B Hugging Face repository is publicly listed, but no model files are cached locally and this session did not attempt a model download. Public listing is not proof that this host can download it or that any credentials/network access are available.

## Inspection commands

Commands run (outputs summarized below):

```sh
uname -a
cat /etc/os-release
python3 --version
python3 -m pip --version
python3 - <<'PY'
import importlib.metadata as m
names=['torch','transformers','datasets','trl','peft','bitsandbytes','unsloth','huggingface-hub','llama-cpp-python']
for n in names:
    try: print(n, m.version(n))
    except m.PackageNotFoundError: print(n, 'not installed')
try:
    import torch
    print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())
except Exception as e:
    print(type(e).__name__, str(e))
PY
nvidia-smi
command -v nvcc
command -v llama-cli
command -v llama-server
df -h .
free -h
git status --short --branch
git rev-parse --show-toplevel
git config --global --list --show-origin
git config --local --list --show-origin
```

A public Hugging Face search was also used to confirm that `Qwen/Qwen3-1.7B` is listed. No weights were fetched.

## Detected versions and state

| Component | Detection |
|---|---|
| OS | Ubuntu 24.04.1 LTS (Noble), WSL2, Linux kernel 6.6.87.2-microsoft-standard-WSL2, x86_64 |
| Python | 3.12.3 at `/usr/bin/python3` |
| pip | 24.0 |
| Git | Installed; global `user.name` and `user.email` are configured, as are GitHub credential helpers. No local Git config was available. The `.git` directory is empty, and Git commands report this directory is not a repository. |
| NVIDIA/CUDA | Confirmed via `nvidia-smi` in a real (non-Codex) WSL shell: Driver Version 582.78, CUDA Version 13.0 (max supported by driver, not an installed toolkit). `nvcc` still not checked in that shell. |
| GPU | GTX 1060 (Max-Q), 6144MiB VRAM, confirmed visible and idle (0% util) in WSL. Compute capability 6.1 (Pascal) — see new compatibility section below. |
| PyTorch CUDA | PyTorch is still not installed, so `torch.cuda.is_available()` has not been run yet. Expect `True` once a compatible build is installed per the compatibility notes below. |
| Relevant Python packages | `torch`, `transformers`, `datasets`, `trl`, `peft`, `bitsandbytes`, `unsloth`, `huggingface-hub`, and `llama-cpp-python`: not installed. |
| llama.cpp | No `llama-cli` or `llama-server` executable found. |
| Qwen3-1.7B | Public Hugging Face repository listing confirmed; no local cache found; host download/authentication was not tested. |
| Workspace filesystem | 366 GiB total, 83 GiB used, 283 GiB available (23% used). |
| RAM | 7.7 GiB total, approximately 6.8 GiB available at inspection; 2.0 GiB swap. |
| Repository content | `implementation_plan_claude.md` only, plus the empty `.git` directory and environment-managed directories. No source code, dependency manifest, or existing benchmark/results were found. |

## Known compatibility issues and cautions

- Python 3.12 may constrain versions/build availability for selected training packages and CUDA extensions. Compatibility should be checked against the chosen cloud image and package versions before pinning a training environment; nothing was installed to test this.
- The project folder is not currently a Git repository despite having an empty `.git` directory. Avoid relying on commit-based tracking until repository metadata is repaired or initialized deliberately.
- The machine reports 283 GiB free disk, but a complete model download plus caches/checkpoints can use substantially more space than the current minimal reconnaissance setup.
- Public model visibility does not establish network reachability, terms acceptance, rate limits, or credential availability from this host.
- Recon run through a sandboxed Codex CLI session can under-report host capability (it missed the GPU entirely). Re-verify any figure that matters before relying on it.

### GPU compatibility (GTX 1060 Max-Q, Pascal, compute capability 6.1)

The GPU itself is confirmed working, but compute capability 6.1 (Pascal) is actively being dropped across the local-inference/training stack as of September–October 2026. This needs explicit version pinning, not default installs:

- **PyTorch — highest risk.** PyTorch wheels built against CUDA ≤12.6 include Pascal (sm_61). CUDA 13.x wheels drop Maxwell/Pascal/Volta; the PyTorch release notice currently identifies 2.14.0 with CUDA 12.6 as the compatible legacy install while that build remains available. A plain `pip install torch` can select a build that cannot use this GPU. **Action:** pin the version and CUDA build and recheck the [current PyTorch compatibility notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) before installing.
- **bitsandbytes (QLoRA) — supported, contingent on the PyTorch pin above.** Hugging Face's bitsandbytes docs list both NF4/FP4 quantization and 8-bit optimizers as supported on NVIDIA Pascal (GTX 10X0 series, P100) or newer. QLoRA itself is not the blocker here; it only breaks if the underlying PyTorch build already dropped sm_61.
- **llama.cpp (Phase G, local deployment) — needs the CUDA 12.4 asset, not the latest release.** llama.cpp's CUDA 13.x builds drop Pascal the same way PyTorch's do, but a separate CUDA 12.4 release asset that explicitly retains compute 6.1 was still being published alongside the 13.x builds as of mid-August 2026. Download that asset specifically for local GGUF inference on this GPU.
- **Driver is already sufficient.** Pascal (compute capability 5.0–6.2) needs driver 570 or newer; this machine is on 582.78.
- **None of this affects Phases D–F (SFT/DPO/GRPO)** — those run on cloud GPUs with no Pascal constraint. It only matters for local dev, local eval, and the final local quantized-inference phase.

## Phase A model smoke check (2026-10-10)

This check was run in the current coding session, which is more restricted than the plain WSL terminal described above:

| Item | Current-session result |
|---|---|
| Python | 3.12.3 |
| `torch` / `transformers` / `huggingface_hub` | Not installed |
| Qwen3 cache / repository `models/` / `checkpoints/` | Not present |
| `nvidia-smi` executable | Present, but NVML reports GPU access blocked by the operating system |
| Model inference | Not started; no model behavior was observed |

The runner `python3 -m evals.manual_smoke` is ready, but its first run is expected to record five inference setup failures until the runtime and weights are supplied. Its JSONL output contains full model/tool trajectories; setup errors are labeled failed and are not presented as model-evaluation data. Every task works in a disposable copy of this repository, so the trivial file-write task cannot modify the real checkout.

## Hackathon cloud baseline environment (Phase 2, 2026-11-04)

The hackathon path uses a separate cloud environment and does not use the local
GTX 1060 for model inference. `configs/tiny_aya_eval.json` pins Python 3.12,
PyTorch 2.11.0, Transformers 5.22.1, Accelerate 1.15.0, and AgentTune 1.1.3.
These are intended cloud-job versions; they are not claimed as installed in the
local WSL virtual environment. Tiny Aya Global is pinned to Hub revision
`af89d219b53ed9b13b8a4645f9c8028973510324` and loaded in FP16 on a CUDA job.

The smoke job installer now uses those exact package pins. Phase 2 added a
separate held-out evaluator and a bundle builder that checks the permanent
held-out order and leaves `train_support.jsonl` out of the job bundle. No
formal baseline or training run was performed in this phase.

Static checks used during implementation:

```sh
python3 -m py_compile training/agenttune/hf_gpu_smoke.py training/agenttune/evaluate_heldout.py training/agenttune/prepare_hf_eval_bundle.py scripts/extract_hf_baseline.py
python3 training/agenttune/prepare_hf_eval_bundle.py --destination /tmp/terminal-slm-eval-check
```

The syntax check, frozen prompt/schema hash check, and held-out bundle staging
completed. The staged bundle contained IDs B005, B010, B015, B020, B025, B030,
B035, B040, B045, and B050, and no support task file. These are implementation
checks only; they are not model evaluation results. The evaluator expects CUDA
and verifies exact runtime versions, so running it locally without its pinned
cloud dependencies is intentionally unsupported.

### Owner actions to unblock Phase A

1. In the WSL environment where `nvidia-smi` sees the GTX 1060, create and activate a project virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.
2. Install a PyTorch CUDA build that still includes Pascal (`sm_61`) support. As of this report, the PyTorch release guidance identifies the CUDA 12.6 build as the legacy option for Pascal and says to pin PyTorch 2.14 while that build is available. The official install command is `python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126`; confirm the current compatibility notice before installing because wheel support changes over time. See [PyTorch's CUDA 12.6/Pascal notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) and [official previous-version commands](https://pytorch.org/get-started/previous-versions/).
3. Install the model runtime with `python -m pip install 'transformers>=4.51.0' huggingface_hub`. The Qwen3 model card says Transformers versions older than 4.51.0 do not recognize the `qwen3` model type; use the [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) as the source of current model-specific instructions.
4. Verify in that same shell before downloading weights:

   ```sh
   nvidia-smi
   python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device")'
   ```

   If CUDA is false or the device is unsupported, stop and resolve the PyTorch/GPU compatibility issue; do not install training packages for this smoke test.
5. If Hugging Face access is available and you accept the model's terms, download the base weights explicitly: `hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B`. The harness itself uses `local_files_only=True` and will not download them implicitly. A local checkpoint path avoids needing credentials in the agent process.
6. From the active environment, first run `python3 -m unittest discover -v`, then run `python3 -m evals.manual_smoke --model-path models/Qwen3-1.7B --max-seconds 600`. The first task includes lazy model loading; `--max-seconds` is per task and the smoke runner reuses the loaded model for later tasks. The default 120 seconds can be too short for a cold load on this machine. Review every raw trajectory and failed verifier before claiming the Phase A smoke gate passes. Do not modify the model weights.

### What to do at the end of each research phase

- **A — Harness:** you supply the compatible local inference runtime and base weights, run the manual smoke tasks, and review trajectory failures. A is ready to exit only after several tasks complete through tool call, observation, and final answer.
- **B — Formal benchmark:** review/approve task definitions, workspace isolation, and objective checkers before the benchmark set is frozen. No model installation beyond A's base inference setup is needed to author it.
- **C — Baseline:** approve a frozen run configuration; run all benchmark tasks with the unchanged base model; retain per-task trajectories, outcomes, latency, and aggregate metrics before training changes anything.
- **D — QLoRA SFT:** provide/choose a cloud GPU allocation, prepare reviewed training examples, and approve a separate pinned training environment and storage plan. Training packages belong in that cloud environment, not the local smoke environment by default.
- **E — Optional DPO:** first review preference-pair data and decide whether baseline failures justify DPO; if not, skip E. If yes, provide/approve the data and cloud run budget.
- **F — Optional GRPO:** first approve a stable benchmark reward and cloud compute plan; if either is missing, skip F. Review the reward and run logs before accepting a checkpoint.
- **G — GGUF/local deployment:** choose the candidate checkpoint and quantization, then validate the artifact and tool behavior in llama.cpp on the target local machine. Install llama.cpp only at this phase; compare against the saved base-model baseline.

## WSL inference attempt update (2026-10-16)

The project owner reports that the local environment now has a `.venv` and the checkpoint at `models/Qwen3-1.7B`. The CLI was first run without `--model-path`, so it selected its default `Qwen/Qwen3-1.7B` identifier. Since `LocalTransformersClient` uses `local_files_only=True`, it searched the Hugging Face cache and failed; it does not look in `./models/Qwen3-1.7B` automatically. For the coding task, `--workspace ./my-project` also selects that separate directory; use `--workspace .` when the current checkout is the repository to edit.

The subsequent smoke run passed `--model-path models/Qwen3-1.7B` and reported all 311 weight shards loaded, but all five tasks then ended with `model_error`, no tool call, and no model output. The traceback identified a client bug: `apply_chat_template` returned a Transformers `BatchEncoding`, while the client treated it as a tensor and accessed `.shape`. The client now requests a dictionary-style encoding, gets the input length from `input_ids`, and calls `model.generate(**encoded, ...)`, including the attention mask. A dependency-free regression test covers this interface. The deprecated `torch_dtype` warning was unrelated.

The code also records exception class and traceback in model events, counts model loading against the per-run deadline, and exposes `--max-seconds`. The real end-to-end smoke gate remains open until these commands pass on the owner’s machine:

```sh
./term-agent --model-path ./models/Qwen3-1.7B --workspace . --max-seconds 600 "Fix the failing authentication tests"
python -m evals.manual_smoke --model-path ./models/Qwen3-1.7B --max-seconds 600
```

The owner reran the suite at `research/results/manual_smoke_20261016T180925Z.jsonl`. All five tasks completed with one successful tool call each. The two reported failures were verifier false negatives: the unittest command exited successfully and printed `OK` to stderr, while its verifier only searched stdout; the marker file contained exactly the requested text but no final newline, while the verifier required one. Updated verifiers check both output streams and accept either exact one-line representation. The saved run therefore qualifies as 5/5 on corrected checks, though a fresh run is recommended to record the corrected result directly.

The first command deliberately targets the current checkout; use a different explicit workspace if that is not the repository you intend to edit. The timeout is per task and includes cold model loading; this code cannot interrupt synchronous model loading, but it will stop before generation if loading consumes the budget. Inspect the new trajectory's traceback if inference still fails. Do not treat any of these as tool/parser failures until a model response is actually recorded.


