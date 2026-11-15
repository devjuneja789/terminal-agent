# Terminal-SLM — Track 3 project record

## Project choice

- **Track:** 3 — Agentic Fine-tuning.
- **Model:** `CohereLabs/tiny-aya-global`, pinned to revision
  `af89d219b53ed9b13b8a4645f9c8028973510324` for the reported experiments.
- **Training/evaluation:** AgentTune for the Cohere action loop and SFT;
  AuditKIT for recorded-episode structural evaluation alongside independent
  executable benchmark contracts.
- **Compute:** Hugging Face GPU Jobs for all model training and inference.
  Local development is CPU-only in the current environment; NVML access is
  blocked and local PyTorch is absent.

## Research question

Does AgentTune post-training on reviewed terminal-use demonstrations improve
Tiny Aya's executable task success and tool-call reliability compared with the
untouched model under a fixed evaluation procedure?

The benchmark is a 50-task deterministic repository set, split permanently into
40 support and 10 held-out tasks. Held-out IDs, patches, outcomes, and rewards
are excluded from all training-data and support-selection paths.

## Completed evidence

1. **Baseline:** untouched Tiny Aya scored 0/10 on the frozen held-out
   evaluation (`research/experiments/EXP-004-tiny-aya-baseline.md`).
2. **Support rollout and data review:** early model rollouts did not produce
   training-eligible examples. Four separately owner-reviewed demonstrations
   were admitted as 15 continuation targets; truthful-failure examples remain
   audit-only.
3. **SFT smoke and diagnostics:** bounded jobs verified labels, gradients,
   optimizer updates, adapter changes and reloads. Several fixed support replays
   remained unsuccessful, motivating a task-start coverage diagnostic.
4. **Frozen matched evaluation (EXP-025):** the selected adapter scored 2/10
   contracts versus 1/10 for the base. Both scored 0/5 on repair/update tasks;
   trace review found the adapter's additional explanation pass factually
   unreliable. No reliable end-to-end gain is claimed.
5. **Later support-only follow-up (EXP-027):** a separate task-start adapter
   scored 2/4 versus 0/4 for the base on four known support tasks. Its two
   successful full-file edits matched reviewed references and passed named and
   regression tests. The other two tasks failed. This adapter was not tested on
   held-out tasks and does not replace EXP-025's frozen candidate or results.

The detailed protocols, artifacts, failure analyses and limitations are linked
from `research/experiments/` and `submission/RESULTS.md`.

## Architecture and data boundaries

AgentTune supplies the model/action loop for the cloud experiments. Terminal-SLM
supplies the five workspace-scoped tools, disposable repository workspaces,
fixed task snapshots, executable validators, result extraction, and provenance
audits. AuditKIT reads actual recorded AgentTune episodes; it does not generate
model outputs or replace task validators.

Only reviewed support targets are used for SFT. The held-out tasks stay fixed
and evaluation-only. Support replay results are labeled as known-task evidence;
they are not presented as generalization. Raw evaluation outcomes are never
converted into training labels.

## Submission scope and remaining owner work

This repository contains the code, configs, provenance, actual run records,
reports and recorded walkthrough. No claim is made that model/data collections
or a public deployment are already published. Before public artifact release,
review the base model's noncommercial terms, choose/confirm the project's own
source/data licensing, publish only the approved payloads, and verify public
links. No further training is recommended for this submission.
