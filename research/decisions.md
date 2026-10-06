# Research decisions

This file records project decisions and their rationale. No implementation or experiment decisions have been made yet beyond the research direction stated in the project brief.

## Initial direction

- **Starting model:** `Qwen/Qwen3-1.7B`.
- **Target behavior:** reliable use of a constrained terminal/tool interface for executable coding tasks.
- **Evaluation:** compare the unmodified model and adapted checkpoints using an executable benchmark with objective success checks.
- **Training:** QLoRA SFT is the planned first training method; DPO and GRPO remain optional and depend on evidence, data, and compute.
- **Compute split:** local machine for development, evaluation, and quantized inference; cloud GPU for training.
- **Current status:** environment reconnaissance and planning only. Phase A has not started.

## Decisions to resolve before implementation

- Confirm/repair Git repository metadata and select branch/history conventions.
- Verify WSL access to the stated GTX 1060 Max-Q or explicitly target CPU-only development for now.
- Choose and pin the runtime/development environment after checking compatibility on the actual target machines.
- Confirm model artifact access and the exact checkpoint/revision before baseline evaluation.
- Define tool boundaries, benchmark task format, success criteria, and result-recording format before generating data or reporting metrics.
