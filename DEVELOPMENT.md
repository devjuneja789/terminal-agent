# Development and environment notes

## Active project setup

Terminal-SLM's reported model experiments use Hugging Face GPU Jobs. The local
checkout is for code, documentation, and CPU-side audits; do not expect local
model inference to work without installing a compatible GPU stack and model
weights. The current local environment did not expose its GPU to NVML, and no
PyTorch or model packages are installed there. No dependencies or weights were
installed as part of this repository review.

The Tiny Aya experiments pinned the model revision and recorded their own job
environments. The final matched evaluation used a Tesla T4 with 16 GiB VRAM,
Python 3.12.16, PyTorch 2.11.0+cu130, Transformers 5.22.1, AgentTune 1.1.3,
Accelerate 1.15.0, TRL 1.7.1, PEFT 0.21.3, bitsandbytes 0.50.2, and Hugging Face
Hub 1.33.0. Use the per-run metadata/config as the source of truth when
reproducing an experiment.

## Latest local inspection

Commands used in the current checkout:

```sh
uname -a
python3 --version
python3 -c 'import importlib.metadata as m; print(m.version("torch"))'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h .
free -h
git config --get user.name
git config --get user.email
```

Detected values:

- OS/kernel: WSL2 Linux, kernel `6.6.87.2-microsoft-standard-WSL2`, x86_64.
- Python: 3.12.3.
- `torch`, `transformers`, `datasets`, `trl`, `peft`, `bitsandbytes`, `unsloth`, and `huggingface_hub`: absent from the active Python environment.
- `nvidia-smi`: installed, but NVML reports GPU access blocked by the operating system. This session cannot verify or use local CUDA.
- RAM: 7.7 GiB total; approximately 6.9 GiB available at inspection.
- Workspace filesystem: approximately 261 GiB free.
- Git author name and email are configured. Their values are intentionally not copied into this document.
- `llama.cpp`: not checked for the current submission because local model deployment is deferred.

The project owner reports a local GTX 1060 Max-Q with 6 GiB VRAM. That hardware is
not exposed to this environment and is not used for the reported Tiny Aya runs.
Do not run inference on CPU as a fallback. Cloud CUDA job metadata is retained
with the real results.

## Checks

The lightweight repository test suite uses Python's standard library:

```sh
python3 -m unittest discover -v
```

AuditKIT is optional for development. Its dependencies are listed in
`requirements-audit.txt`; installing them is not required to inspect the saved
reports. To evaluate the existing final records without model inference:

```sh
python3 -m evals.auditkit_report \
  research/results/tiny-aya-final-heldout/20261115T071710Z.jsonl \
  --output-prefix submission/auditkit/final
```

The audit command reads saved records; it does not contact a model endpoint or
rerun task validators. For exact experiment reproduction, follow
[submission/REPRODUCING.md](submission/REPRODUCING.md). Job launchers may incur
GPU charges and require Hugging Face authentication; never put tokens into
tracked files, shell history, or logs.

## Compatibility and safety notes

- Training/inference packages are cloud-pinned; local absence is intentional.
- The recorded GPU jobs used a Tesla T4 and CUDA-enabled PyTorch. Their versions
  must be taken from each job's metadata rather than inferred from the local
  environment.
- Tiny Aya's base model terms and attribution requirements apply to derived
  adapters. See the release review before publishing model or data artifacts.
- Shell tool restrictions are application guardrails, not an OS sandbox. Run
  arbitrary repositories only in an appropriately isolated environment.
- The final matched benchmark adapter and the later support-only task-start
  adapter are distinct experiments. EXP-027 was not run on held-out tasks and
  must not be described as the held-out candidate's measured result.

## Recommended workflow

1. Create/activate a local virtual environment for repository checks if desired;
   no model stack is required for the core unit tests.
2. Run the complete test suite and inspect saved JSONL, metadata, audits, and
   experiment notes before changing evaluation code.
3. For any cloud job, review its fixed config and data manifest, confirm the
   intended support/evaluation split, and supply credentials through the HF Job
   secret mechanism.
4. Keep actual outputs, failure categories, and support-only versus held-out
   evidence distinct. Do not turn training loss or hand-written references into
   model task-success claims.
