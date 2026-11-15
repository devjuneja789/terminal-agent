# Terminal-SLM

**Track 3: Agentic Fine-tuning.** Terminal-SLM studies whether reviewed
supervised demonstrations can improve a small model's reliability when using
workspace-scoped terminal tools on executable coding tasks.

## Current result

The frozen, matched held-out evaluation compared the untouched
`CohereLabs/tiny-aya-global` model with a support-selected adapter. The base
passed 1/10 benchmark contracts and the adapter passed 2/10. Neither completed
a code-repair task (0/5 each), and trace review found the adapter's additional
pass contained an inaccurate explanation. We do **not** claim reliable
end-to-end improvement.

A later, separate support-only task-start experiment passed 2/4 tasks with its
adapter versus 0/4 with the base. Two file edits were independently checked
against reviewed references and tests; two tasks still failed. This is
exploratory support evidence, not held-out performance, and that adapter was
not evaluated on the held-out set. See [EXP-027](research/experiments/EXP-027-task-start-curriculum-plan.md).

See the [matched evaluation](submission/RESULTS.md), [trace review](submission/TRACE-REVIEW.md),
[reproduction guide](submission/REPRODUCING.md), and [recorded walkthrough](submission/demo.html).

## Research question and method

Does AgentTune supervised post-training on reviewed terminal-use demonstrations
improve Tiny Aya Global's executable task success and tool-call reliability
compared with the untouched model under a fixed evaluation procedure?

The active experiment path is:

```text
CohereLabs/tiny-aya-global
    → AgentTune Cohere-action terminal loop
    → fixed 50-task repository benchmark (40 support / 10 held-out)
    → matched base evaluation
    → reviewed support-only QLoRA SFT
    → frozen adapter evaluation + AuditKIT trace audit
```

Inference and training run on Hugging Face GPU Jobs. The local machine is used
for repository work and CPU-only checks. The project does not use held-out task
records, outcomes, or rewards as training data. DPO, GRPO, GGUF conversion, and
local model deployment were not part of this submission.

## Architecture

The cloud evaluation runner uses AgentTune for the model/action loop and
trajectory capture. Terminal-SLM supplies isolated repository snapshots,
five workspace-scoped tools (`shell`, `read_file`, `write_file`, `grep`, `git`),
executable task validators, frozen split definitions, and audit scripts. The
complete trajectories are checked by independent harness metrics and AuditKIT.

The repository also contains the earlier standalone tool harness and CLI in
`agent/`, `tools/`, `model/`, and `cli.py`. The current reported cloud
experiments use `training/agenttune/`; the standalone local Transformers path
is not used to produce the reported results.

## Experiment plan

- **A — Agent/tool harness:** workspace-scoped tools, bounded loop, trajectory capture, and CLI implemented. Cloud experiments use the AgentTune Cohere-action integration.
- **B — Executable benchmark:** 50 deterministic repository tasks; fixed 40-task support and 10-task held-out split.
- **C — Baseline:** untouched Tiny Aya scored 0/10 on the frozen held-out benchmark.
- **D — Supervised fine-tuning:** adapter training, label/gradient/update/reload checks, paired held-out evaluation, and a later support-only task-start diagnostic completed. Results and limitations are documented; no reliable repair gain was established.
- **E — DPO:** not run; no need established for this submission.
- **F — GRPO:** not run; deferred.
- **G — Quantized local deployment:** not run; deferred. The chosen inference path is cloud GPU evaluation.

## Reproduce and inspect

Core repository checks need Python's standard library:

```sh
python3 -m unittest discover -v
```

To independently inspect the completed held-out records with AuditKIT, install
the audit dependencies from `requirements-audit.txt`, then run:

```sh
python3 -m evals.auditkit_report \
  research/results/tiny-aya-final-heldout/20261115T071710Z.jsonl \
  --output-prefix submission/auditkit/final
```

This audits saved episodes and does not run model inference. Cloud job launchers
in `scripts/` require Hugging Face access, credentials supplied securely to the
job, and billable GPU resources. Review their configs and the reproduction guide
before launching jobs.

## Hardware and availability

- **Local development host:** WSL2 Ubuntu, Python 3.12.3, about 7.7 GiB RAM, and a workspace filesystem with about 261 GiB free at the latest check. The host GPU is a GTX 1060 Max-Q with 6 GiB VRAM, but NVML access is blocked from the current environment. PyTorch and the model-training packages are not installed locally; local CUDA inference was not attempted.
- **Cloud experiments:** Hugging Face GPU Jobs. The matched held-out run used a Tesla T4 (16 GiB); the recorded job environment used Python 3.12.16, PyTorch 2.11.0+cu130, Transformers 5.22.1, AgentTune 1.1.3, and Accelerate 1.15.0. Training and inference configuration are recorded with each run.
- **Model:** Tiny Aya Global, pinned to revision `af89d219b53ed9b13b8a4645f9c8028973510324` in the reported experiments.

See [DEVELOPMENT.md](DEVELOPMENT.md) for inspection commands and compatibility notes.

## Repository map

- `agent/`, `tools/`, `model/`, `cli.py` — standalone workspace tool harness and CLI.
- `evals/benchmark/` — task contracts, snapshots, fixed split, and evaluator.
- `training/agenttune/`, `configs/`, `scripts/` — cloud training/evaluation integration and reproducible launch/extraction utilities.
- `data/` — reviewed support examples, admission decisions, and provenance.
- `research/experiments/` — protocols, results, failure analysis, and design decisions.
- `research/results/` — actual run outputs; raw GPU job console logs are excluded by `.gitignore`.
- `submission/` — results, reproduction instructions, trace review, and a recorded demo.

## Limitations

The benchmark is small, shares one synthetic mini-application, and uses narrow
validators. Task-level separation is not repository-level separation. Contract
scores do not guarantee factual answer quality; the trace review documents a
counterexample. Support results are from known tasks and are not generalization
evidence. The shell layer is a workspace-scoped guardrail, not an OS security
sandbox. No production safety or autonomous coding guarantee is claimed.
