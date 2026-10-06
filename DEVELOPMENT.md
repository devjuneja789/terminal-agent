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

- **PyTorch — highest risk.** PyTorch wheels built against CUDA ≤12.6 still include Pascal (sm_61). Starting with the PyTorch 2.15 release, CUDA 12.6 wheels — the last ones covering Pascal — stop being published, because CUDA 13.x wheels only support Turing (sm_75) and newer. A plain `pip install torch` will grab the newest wheel and silently produce a build that can't see this GPU (`CUDA capability sm_61 is not compatible... supports sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`). **Action:** always install a specific version + CUDA build, e.g. `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126`, and never let a later `pip install -U torch` replace it with an unpinned build.
- **bitsandbytes (QLoRA) — supported, contingent on the PyTorch pin above.** Hugging Face's bitsandbytes docs list both NF4/FP4 quantization and 8-bit optimizers as supported on NVIDIA Pascal (GTX 10X0 series, P100) or newer. QLoRA itself is not the blocker here; it only breaks if the underlying PyTorch build already dropped sm_61.
- **llama.cpp (Phase G, local deployment) — needs the CUDA 12.4 asset, not the latest release.** llama.cpp's CUDA 13.x builds drop Pascal the same way PyTorch's do, but a separate CUDA 12.4 release asset that explicitly retains compute 6.1 was still being published alongside the 13.x builds as of mid-August 2026. Download that asset specifically for local GGUF inference on this GPU.
- **Driver is already sufficient.** Pascal (compute capability 5.0–6.2) needs driver 570 or newer; this machine is on 582.78.
- **None of this affects Phases D–F (SFT/DPO/GRPO)** — those run on cloud GPUs with no Pascal constraint. It only matters for local dev, local eval, and the final local quantized-inference phase.


