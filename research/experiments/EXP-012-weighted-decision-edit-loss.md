# EXP-012: Fixed decision/edit weighted completion loss

## Hypothesis

EXP-011 found correct prefix/label boundaries but wrong tool choices on every
edit target. Conditional token accuracy above 96% on those actions hid errors
in `write_file`, changed code, and JSON closure. Increasing these tokens' loss
contribution may improve executable task completion on the approved support
examples. This is an overfitting diagnostic, not a generalization experiment.

## Fixed configuration and procedure (declared before execution)

- Fresh pinned Tiny Aya base, revision `af89d219b53ed9b13b8a4645f9c8028973510324`.
- Same four owner-approved references and 16 action targets as EXP-010/011.
  Rows SHA-256: `ef2ef134f4b30dc822879e85d238e206c1c6ef9e949d83f49c60c21c75df33ae`.
- Exactly 64 updates/four epochs; seed 42, batch 1, accumulation 1, LR 1e-4
  with linear decay, LoRA rank 8/alpha 16 on q_proj/v_proj, NF4 frozen base,
  FP16 compute and FP32 trainable adapters. Runtime pins unchanged.
- Only objective changes: tool-name tokens **8x**, changed-code tokens **8x**,
  trailing JSON closure tokens **4x**. Other completion tokens and EOS **1x**;
  prompt/context tokens remain masked with weight zero. Overlaps take the
  maximum weight. Normalize by summed active weights per batch.
- Changed code comes from a deterministic character diff between each approved
  write and its preceding full-file read observation. Map escaped JSON offsets
  to actual tokenizer offsets, failing on ambiguous or missing spans. Preserve
  every approved prompt, completion and token boundary. Save the token weights,
  span positions and source identities with training artifacts.
- Verify unit weights reproduce ordinary completion CE values and gradients
  using synthetic tensors locally. All model training/inference runs on an HF
  T4 GPU. Log weighted loss and ordinary completion NLL separately; weighted
  loss is not directly comparable to EXP-010's training loss.
- Run the existing paired base/adapter replay on B021/B026/B041/B046 with the
  same tasks, prompts, schemas, quantization, greedy generation and validators.
  No gold targets enter the inference bundle. Compare actual edits, task
  success and call validity with fresh base and archived EXP-010 results.
- No held-out access, new training text, adaptive weight sweep or longer run.
  Single-seed, four-example results cannot establish held-out improvement.

## Results

### Numerical startup correction

First job `6af882df23985fb9cd823f21` used the default FP16 AMP starting
scale 65536. Its live log recorded nonfinite gradient norms and skipped
updates at steps 3 and 10 (the learning-rate schedule also stayed unchanged
on these steps). It was cancelled, without replay or accepting its adapter.
The ignored raw console log is retained locally. This is a failed attempt.

The rerun fixes only numerical scaling: initialize the existing FP16
GradScaler at 8192 before optimizer construction, retaining its growth/backoff
settings and the same unscaled objective/optimizer. Record before/after scaler
state and abort immediately on any skipped update. No weight, data, step,
learning-rate, precision or evaluation changes are made. This differs from
EXP-010's default AMP startup scale and is an explicit comparison limitation;
the all-ones objective equivalence check still applies mathematically.

### Completed training

Corrected job `6af8865329dd05ff0367796e`, run `20261114T160511Z`, completed
64 updates/four epochs with no skipped steps and finite loss/gradient norms.
Runtime was 199.0853 seconds; peak CUDA allocation 7,900,909,056 bytes.
Adapter weights changed and reload passed. Staged config SHA-256:
`ce3c22370c47a7629a18e82d8093782239e16cbf2227e5afafd2a097c2b24637`.
The maximum training length was 2345 tokens, below the unchanged 4096 cap.

| Epoch | Mean weighted batch loss | Mean ordinary completion NLL |
|---|---:|---:|
| 1 | 0.76160 | 0.83741 |
| 2 | 0.51537 | 0.58682 |
| 3 | 0.36147 | 0.44165 |
| 4 | 0.31721 | 0.39754 |

Aggregate weighted training loss: 0.48891097. The ordinary loss in epoch 4 is
close to EXP-010's 0.39443; the weighted columns cannot be directly compared
with that experiment's unweighted objective. These are training batch means.

All 16 token audits match character spans recomputed from the unchanged
approved text. All three writes have changed-code spans; every EOS label has
weight 1. Original AMP scale 65536 and corrected scale 8192 are recorded with
identical growth/backoff settings. Raw result, 64-step history, 16 token audits,
analysis and source/log hashes are under
`research/results/tiny-aya-weighted-sft/20261114T160511Z.*`.

Private adapter: `devjuneja789/terminal-slm-tiny-aya-weighted-64-v1`, commit
`1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0`, subfolder
`20261114T160511Z/adapter`.

### Paired replay

Completed job `6af887c823985fb9cd824409`, run `20261114T161125Z`, uses config SHA-256
`dee750d9ebb9cd507af57711a4ab830f0f4f1b7d20eeccbef67f35ffef37579f`.
Configuration: `configs/tiny_aya_weighted_sft.json`.
Launcher: `bash scripts/run_tiny_aya_weighted_sft.sh`.

All eight full trajectories were recovered. Recomputed aggregates match the
saved summary. The replay configuration differs from EXP-010 only in adapter
identity/lineage and config ID; quantization and effective tokenizer match.
All four first prompts equal both the paired arm's prompt and the approved
reference prompt. Data, contracts, schemas and inference controls remain fixed.

| Metric | Fresh base | Archived unweighted 64-step | Weighted 64-step |
|---|---:|---:|---:|
| Task contract passes | 0/4 | 0/4 | 0/4 |
| Executed write_file calls | 0 | 0 | 3 |
| Tasks with changed source files | 0 | 0 | 2 |
| Successful repairs | 0 | 0 | 0 |
| Argument/schema validity | 10/12 (83.33%) | 12/13 (92.31%) | 19/19 (100%) |
| Invalid call rate | 16.67% | 7.69% | 0% |
| Extra-action/prose violations | 6 | 0 | 0 |
| Tool selection accuracy | 50.00% | 44.44% | 63.64% |
| Average executed tool steps | 2.00 | 2.25 | 3.75 |
| Average model turns | 3.75 | 4.00 | 5.75 |
| Generated tokens across four tasks | 3840 | 673 | 965 |
| Average prompt-plus-output tokens | 7572.50 | 7294.50 | 11892.25 |
| Average agent latency, seconds | 60.29 | 15.61 | 23.50 |

Archived results are from EXP-010's separate paired job. Single seed, fixed arm
ordering, temporary-path token differences and different generation lengths
limit causal/speed comparisons. Required-tool accuracy can penalize additional
source reads; inspect traces alongside this metric. Schema validity counts
attempted terminal calls, including calls blocked by the repeat guard; it is
not a code-correctness measure.

### Factual trace audit

- **B021:** reads only lines 1–4 (docstring/import), then executes write_file
  with `page_size = 1 if page_size == 0 else page_size` as the entire module.
  The named test fails on a NameError. It reads the broken file and proposes
  the same write again; the repeat guard blocks it and requests a model finish.
  Its final answer falsely claims correctness. One write actually executed.
- **B026:** reproduces the expected failing test and reads the full module,
  then repeats the read. No write executes. The model's recovery finish
  falsely claims the test passes after a configuration change.
- **B041:** reads only lines 1–2, then overwrites the module with the desired
  constant and an undefined `request_timeout()` call. A second write removes
  whitespace but leaves the NameError. Two writes execute; neither preserves
  the module. It later repeats the same write and enters finish recovery.
- **B046:** runs the requested intentionally failing test, observes `10 != 25`
  and reads the full module. Its answer restates the mismatch but does not
  identify the unconditional constant return/ignored environment variable.
  The intentional nonzero test exit is valid evidence, not the rejection cause.

All four weighted episodes needed repeat-recovery prompting; all final answers
came from the model, not a fabricated harness answer. File replacements occurred
only in disposable benchmark workspaces. None modified validators/tests.
No replay trajectory is admitted as training data or labeled owner-approved.

Full records, metadata, summary, independent analysis and log provenance:
`research/results/tiny-aya-weighted-replay/20261114T161125Z.*`.
Local verification: 106 tests passed with three optional Torch skips under
system Python; all nine weighting tests passed in the Torch virtual environment.

## Conclusion and next experiment

The objective produced a narrow behavior change: the model now executes writes
on two edit tasks and has no invalid terminal calls in this replay. It still
completes zero tasks. Wrong write contents delete required module code, and
unsupported completion claims and repeated actions persist. This is protocol
and tool-choice progress, not successful coding, generalization or a trained
agent milestone. Stop at the fixed run; do not extend epochs or tune held-out tasks.

Before further training, compare the weighted adapter's per-action predictions
under the approved full-file observations with its live partial-file histories.
Check whether it can reconstruct a complete, minimally changed module when the
correct source is supplied. That separates failure to learn the write target
from failure to gather the source needed to preserve it. Keep data/evaluation
fixed for that diagnostic. No further experiment was launched here.
