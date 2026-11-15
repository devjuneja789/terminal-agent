# Terminal-SLM: Learning Tool Actions Is Not the Same as Completing Tasks

**Track 3 — Agentic Fine-tuning**

Terminal-SLM tests whether reviewed supervised demonstrations can improve a small model's reliability on executable terminal tasks. The project uses CohereLabs Tiny Aya Global, AgentTune for its model/action loop and fine-tuning, and AuditKIT to audit recorded episodes. Five workspace-scoped tools operate on isolated repository snapshots; executable validators determine task success.

## Evaluation

The benchmark contains 50 deterministic tasks: 40 support tasks and 10 held-out tasks. The split was fixed before training-data preparation. Only reviewed support demonstrations were admitted for SFT; held-out tasks, outcomes, patches, and rewards were excluded from training.

We first evaluated the untouched base model. After support-only fine-tuning and support-based candidate selection, we ran a matched comparison using the same held-out tasks and inference configuration. We retained complete trajectories and independently checked outcomes against the task contracts.

## Results

The untouched base passed **1/10** held-out task contracts; the selected adapter passed **2/10**. However, both completed **0/5** code-repair or update tasks. Trace review also found the adapter's additional contract pass contained an inaccurate explanation. We therefore do not claim a reliable end-to-end improvement.

A later, separate task-start experiment scored **2/4** versus **0/4** for the base on four known support tasks. Its two successful full-file edits matched reviewed references and passed the named tests plus a regression test. The other two tasks failed. This follow-up was not evaluated on held-out tasks and is not evidence of generalization.

## What we learned

The adapter learned some actions when given reviewed contexts, but those gains did not translate into dependable completion from fresh task conversations. Tool-choice accuracy and training loss alone were insufficient indicators of success. The small benchmark, shared synthetic repository, and narrow validators also limit the conclusions.

This negative result is useful: it separates conditional action learning from full-agent task completion and shows why trajectory audits and executable checks matter. DPO, GRPO, and local quantized deployment were outside this submission's scope.

## Artifacts

- Code: [GITHUB_REPOSITORY_URL]
- Models and adapters: [HF_MODEL_COLLECTION_URL]
- Datasets and evaluation traces: [HF_DATASET_COLLECTION_URL]
- Results and reproduction notes: [GITHUB_REPOSITORY_URL]/tree/main/submission

The model collection should link to the pinned base model and versioned adapters; it must not rehost the gated base weights. Dataset entries should label held-out records as evaluation-only. Tiny Aya is licensed CC-BY-NC-4.0 and requires following Cohere Labs' acceptable-use policy. Replace the placeholders only after the repositories and collections are public.
