# Terminal-SLM

**Can a small language model learn to use terminal tools reliably?**

Terminal-SLM explores this question by training **CohereLabs Tiny Aya Global**
on reviewed examples of repository tasks. An agent reads files, searches code,
runs tests, and makes edits in a temporary workspace. Executable checks verify
whether it actually completed the task.

Built for **Track 3: Agentic Fine-tuning**, using **AgentTune** for training and
agent execution, and **AuditKIT** for auditing saved trajectories. All model
training and inference ran on Hugging Face GPUs.

## Results

The project demonstrates a working training and evaluation pipeline, with
limited task success. The two experiments below use different adapters and
must be interpreted separately.

| Experiment | Untouched base | Trained adapter | What the result means |
|---|---:|---:|---|
| Matched held-out evaluation (EXP-025) | 1/10 | 2/10 | Benchmark contract passes. The extra adapter pass contained an inaccurate explanation; both models completed 0/5 code repairs. |
| Later replay on known support tasks (EXP-027) | 0/4 | 2/4 | Two successful edits matched reviewed references and passed tests. This adapter was not evaluated on held-out tasks. |

**These results do not establish reliable autonomous coding or improvement on
unseen repositories.** They show that learning individual tool actions does
not guarantee completion of a full task.

Read the [full results](submission/RESULTS.md), [trace review](submission/TRACE-REVIEW.md),
and [EXP-027 report](research/experiments/EXP-027-task-start-curriculum-plan.md).
The [recorded walkthrough](https://terminal-agent-l6b4.onrender.com) shows actual saved runs.

## How it works

```text
User task → Tiny Aya / AgentTune → tool action → isolated workspace
                    ↑                               ↓
                    └──────── tool observation ─────┘
                                  ↓
                        final answer + task validation
                                  ↓
                        saved trajectory + AuditKIT audit
```

The agent has five tools: `shell`, `read_file`, `write_file`, `grep`, and `git`.
Tools check file paths, limit execution time and output size, filter environment
variables, and log commands. These are application guardrails; arbitrary shell
commands still require a container or VM for security isolation.

The benchmark contains **50 deterministic tasks** across navigation, code and
configuration discovery, bug fixes, test fixes, and failure explanation. Its
split was fixed before training: **40 support tasks and 10 held-out tasks**.
Tasks share a small synthetic repository, so this split does not measure
transfer across unrelated repositories.

Training used reviewed support demonstrations with **QLoRA supervised
fine-tuning**. The selected held-out adapter used 15 corrective action targets;
the later support experiment used 31 approved targets including task starts.
Held-out tasks and evaluation trajectories were excluded from training.

## Start here

- [Reproduction guide](submission/REPRODUCING.md) — dependencies, pinned settings, and cloud commands.
- [Benchmark documentation](evals/benchmark/README.md) — tasks, validation, metrics, and split rules.
- [Development notes](DEVELOPMENT.md) — local environment and compatibility details.
- [Models and adapters](https://huggingface.co/collections/devjuneja789/terminal-slm-models-and-adapters) — the Tiny Aya base model, the evaluated LoRA adapter, and the later EXP-027 adapter (support tasks only).
- [Data and evaluation evidence](https://huggingface.co/collections/devjuneja789/terminal-slm-data-and-evaluation) — approved support targets, fixed benchmark tasks, recorded trajectories, validation outputs, and audit reports.

Run the core tests with Python 3.12; model packages and a GPU are not required:

```sh
python3 -m unittest discover -v
```

For the optional AuditKIT audit, install the dependencies in
`requirements-audit.txt`, then evaluate the saved records:

```sh
python3 -m evals.auditkit_report \
  research/results/tiny-aya-final-heldout/20261115T071710Z.jsonl \
  --output-prefix submission/auditkit/final
```

This command audits existing episodes. Reproducing model training or inference
requires Hugging Face access and paid GPU resources; follow the reproduction
guide before launching a job. The reported experiments used a Tesla T4 with
16 GiB VRAM. Local CUDA inference is not required.

## Repository layout

| Path | Contents |
|---|---|
| `tools/`, `agent/`, `model/`, `cli.py` | Workspace tools and the standalone agent harness |
| `training/agenttune/` | Active cloud training and agent integration |
| `evals/benchmark/` | Fixed task definitions, repository snapshot, and validators |
| `data/` | Support demonstrations, reviews, and training admission records |
| `configs/`, `scripts/` | Experiment settings and launch, extraction, and audit utilities |
| `research/experiments/`, `research/results/` | Experiment notes and actual outputs |
| `submission/` | Results, reproduction guide, article, and recorded walkthrough |

The reported runs use the AgentTune cloud integration. The standalone local
CLI is retained as development code. DPO, GRPO, GGUF conversion, and local model
deployment were not included in this submission.

## Licensing

Source code and tests use the [MIT License](LICENSE). Curated data, evaluation
traces, and Tiny Aya adapters use CC-BY-NC-4.0 and remain subject to Cohere Labs'
acceptable-use policy. See [license scope](LICENSES.md). Base-model weights are
not redistributed.
