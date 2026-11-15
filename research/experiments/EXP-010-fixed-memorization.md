# EXP-010 — Fixed four-epoch memorization diagnostic

Status: 64-step training and paired replay complete; trace audit complete.

Training job: `6af8752229dd05ff036765b0` on HF T4, 25-minute cap.
Staged config SHA-256: `06e3ba2bfc166937738e72301ec47040b8fc4c702269d37d1da3f65aa0d894dc`.
Dataset row hash: `ef2ef134f4b30dc822879e85d238e206c1c6ef9e949d83f49c60c21c75df33ae`.
All 91 local tests and syntax/diff checks passed before launch.

## Question and stopping rule

Can the approved terminal action sequences be learned after full exposure to
the small reference set? EXP-009's eight-step adapter had seen only half an
epoch and completed 0/4 replay tasks. This run uses exactly 64 optimizer steps
(four epochs at batch size one over 16 action targets). No adaptive extension,
task edits, prompt tuning, or new training data are permitted in this run.
The experiment ends after training and one paired four-task replay, regardless
of outcome. It does not establish generalization.

## Training configuration

- Start from untouched CohereLabs/tiny-aya-global revision
  af89d219b53ed9b13b8a4645f9c8028973510324, not the eight-step adapter.
- Use only approved G-B021-v1, G-B026-v1, G-B041-v1, G-B046-v1. The 16 action
  targets and row hash must match EXP-008/009. Pending and rejected examples
  remain excluded. No held-out records are bundled.
- NF4 double quantization, FP16 compute, FP32 trainable LoRA parameters,
  rank 8, alpha 16, dropout zero, q_proj/v_proj targets, seed 42, learning rate
  1e-4, linear decay over 64 steps, no warmup, max gradient norm 1.0, sequence
  cap 4096, gradient accumulation one, and completion-only loss. The longer
  linear schedule changes the per-step learning-rate schedule relative to the
  eight-step smoke. Other model/trainer settings are preserved.
- Pin the successful smoke's runtime, including PEFT 0.21.3, datasets 5.2.0,
  bitsandbytes 0.50.2 and TRL 1.7.1. Use HF T4 only.
- Before training, verify both masked context and nonempty completion labels
  for all 16 targets. Save actual mask counts. Refuse truncation.
- Record every logging step, final global step/epoch, skipped optimizer steps,
  finite aggregate loss, changed adapter weights, and successful adapter reload.
  Replay cannot launch if 64 steps/four epochs did not finish or updates skipped.

## Artifact and replay procedure

Training has a 25-minute HF job cap. Save the new adapter, config, runtime
metadata and full training history in a separate private repository. Capture
the exact upload commit returned by Hugging Face. Emit full metadata in bounded
base64 chunks, so long loss histories survive console limits. Recover local
JSON and JSONL artifacts before proceeding.

Launch a separate inference job (30-minute cap) containing only B021/B026/B041/
B046 task records, their snapshot and evaluator code. It contains no reference
answers or training targets. Validate the training-result lineage and pin the
new adapter commit/subfolder. Replay a fresh NF4 base and the new adapter using
the exact EXP-009 inference controls, including JSON observations and canonical
tokenizer-equivalence checks. Recover all eight complete trajectories, contract
scores, validators, changes, tool validity, tokens and latency.

Local results go in `research/results/tiny-aya-memorization/` and
`research/results/tiny-aya-memorization-replay/`. Preserve failed attempts and
report actual findings. Loss reduction alone does not satisfy the diagnostic:
inspect whether write_file executes, edits pass validation, and the explanation
identifies the ignored environment variable. Contract results require trace
audit. No generated trajectory becomes approved training data automatically.

## Results

Training run `20261114T145152Z` completed 64 optimizer steps and epoch 4.0,
without skipped updates. All 16 target-mask checks passed. Training runtime was
192.7457 seconds and aggregate loss 0.5350136790366378. Mean batch loss by
epoch was 0.79236, 0.53090, 0.42236, and 0.39443. These are training-set batch
means, not task-success measures. Adapter weights changed and reload passed.

The actual result and 64-step history were losslessly recovered under
`research/results/tiny-aya-memorization/20261114T145152Z.*`.
Private artifact repo: `devjuneja789/terminal-slm-tiny-aya-memorization-64-v1`,
commit `270d688c111f3b01f72db6691d770e6f6cec024f`, adapter subfolder
`20261114T145152Z/adapter`. The short post-training probe still proposed a
malformed shell/sed JSON action; no behavior gain is inferred from lower loss.

The workflow validated the result and launched replay job
`6af8769429dd05ff0367677f` with config SHA-256
`bc4eb34b689c5d2cd3a94cba850077151570ebacb459c1749657246cab027a90`.

### Training-length instrumentation correction

Raw training metadata reports `max_training_tokens: 2`. This was a reporting
and preflight bug: Transformers 5 returned a BatchEncoding and `len()` counted
its two fields. Preserve that raw value. The adjacent `.analysis.json` derives
the actual maximum collated length from recorded masked plus target tokens:
2345. All 16 recorded batches are strictly below the 4096-token cap, with
nonempty completion labels. The evidence supports no truncation in this run,
but the original length preflight was ineffective. The helper is corrected
for future runs to count `input_ids` explicitly; a regression test covers an
encoding with 4100 tokens and two fields. No weight or result artifact was
silently rewritten, and no extra training run was launched for this correction.

### Paired replay

Replay run `20261114T145802Z` completed all eight episodes on the HF T4. Raw
JSONL, summary JSON/Markdown, metadata and source-log provenance are in
`research/results/tiny-aya-memorization-replay/20261114T145802Z.*`.
Recomputed metrics match the console summary. Both arms have equal first
prompts, and the adapter's first prompt for each task exactly matches its
approved reference prompt. This rules out a changed initial instruction as
the explanation for the failed first actions. It does not prove token-level
training-label alignment; that needs a separate diagnostic if pursued.

| Metric | Fresh base NF4 | 64-step adapter NF4 |
|---|---:|---:|
| Contract passes | 0/4 | 0/4 |
| Verified code edits | 0 | 0 |
| Executed write_file calls | 0 | 0 |
| Argument/schema validity | 83.33% (10/12) | 92.31% (12/13) |
| Invalid tool-call rate | 16.67% | 7.69% |
| Extra-action/prose protocol violations | 6 | 0 |
| Tool selection accuracy | 50.00% | 44.44% |
| Average tool steps | 2.00 | 2.25 |
| Average model turns | 3.75 | 4.00 |
| Total generated tokens across four tasks | 3840 | 673 |
| Average prompt-plus-output tokens | 7570.25 | 7294.50 |
| Average agent latency, seconds | 58.05 | 15.61 |

Latency is descriptive: fixed arm ordering, different output lengths, and only
four tasks preclude a general speed claim. Tool selection uses the existing
required-versus-unexpected metric and can penalize an additional source read;
it should be read alongside the trace, not as a semantic correctness oracle.

### Trace audit

- B021: both stop on malformed shell/sed JSON, without executing any tool.
  The adapter's output is shorter but still has invalid escapes/missing braces.
- B026: the adapter emits one action per turn, runs the failing test and
  searches the setting, then reads only source lines 1–2 (the docstring).
  It runs `import shop; print(shop.config.__dict__)`, gets an AttributeError,
  and misattributes that import failure to the configuration file. It does
  not edit anything. The base instead falsely claims a successful repair.
- B041: both search for literal task-like text and run the failing test.
  The adapter finishes by saying the default is unknown, despite the test
  observation exposing the mismatch. No write occurs and the validator fails.
- B046: the adapter eliminates the base's invalid start_line=0 attempt and
  produces schema-valid actions. It runs the expected failing test and reads
  the function, but the final answer only restates 10 versus 25. It does not
  identify the function's unconditional constant return/ignored environment
  variable. This is an explanation failure, beyond strict keyword checks.
  The intentional test failure is valid evidence and is not itself rejection.

All tasks leave source files unchanged. Every replay record remains excluded
from training targets; this assistant trace audit does not imply owner signoff.

## Conclusion and next decision

The fixed four-epoch run learned a limited formatting/stopping change: fewer
invalid calls, zero extra-action/prose violations, and shorter generations.
It did not reproduce a single complete approved task or learn to execute an
edit under this configuration. Lower training loss is not task memorization.

Stop at the declared 64 steps and paired replay. Before increasing epochs or
expanding data, inspect per-action exact matches, the critical tool-name tokens,
and training/inference label boundaries on these same approved prompts. That
diagnostic can distinguish inadequate fitting from objective/tokenization
problems. No further training, data generation or held-out evaluation ran here.
