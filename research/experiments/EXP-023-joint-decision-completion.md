# EXP-023: Joint first-decision and complete-action learning

## Fixed protocol before execution

The user authorizes running/submitting the next training job. EXP-022 improved
write-name selection but did not learn finish and degraded edit/JSON quality.
Test a joint objective before another agent replay. No held-out access, new
demonstrations, automatic admission or model/prompt/schema changes.

Same 15 admitted targets, four sources and immutable row hash; fresh pinned
Tiny Aya base, rank-8 q_proj/v_proj LoRA, 60 updates/four epochs, seed 42,
NF4/FP16 compute, FP32 trainable parameters, LR 1e-4/linear, clipping 1, fixed
AMP scale 256, checkpointing and all-15-batch finite-gradient preflight.
Save to a separate private artifact repository. T4-small, training cap 25m.

Objective: 0.75 times cross-entropy at the first token overlapping the native
tool_name value plus 0.25 times EXP-020's normalized weighted completion loss.
The latter retains positive loss on every target token, including code, JSON
and both native closings. This shifts emphasis from easy tool-name continuation
tokens to the first decision while restoring argument/closure supervision.
The mixture weights are loss coefficients, not claims about gradient attribution.
All labels/text/templates remain unchanged. Record effective coefficients;
GPU dummy logits must verify prompt masking, positive completion gradients and
stronger first-decision gradients. Abort on nonfinite gradients, skipped updates,
changed lineage, truncation, or failed save/reload. No adaptive budget extension.

After training, repeat the same 15 exact contexts on fresh base and new adapter:
30 predictions, greedy seed 42, native stops [3,6,261001], 256-token cap, no
trimming, no terminal tools. Require all 45 context/prediction records and exact
training-token/weight reproduction. Compare to EXP-021 and EXP-022, with repeated
base-control parity. Inference T4-small capped at 30m. No full-agent replay
follows automatically.

Report correct valid tool choices, read/write/finish counts, exact actions,
schema validity, actual proposed file content, teacher scores and termination.
Primary known-context learning criterion remains 15/15 valid correct choices;
secondary nonregression goals are at least 3/15 exact actions and 15/15 valid
schemas. These are training-context criteria, not generalization or repaired
repositories. Stop after the fixed experiment regardless of outcome.

## Status

Local checks pass: 153 tests run (150 pass, three optional Torch checks skip),
including three new mixture/staging/validation tests. Shell syntax, compilation
and `git diff --check` pass. HF Job
[6af94f0d23985fb9cd82f91d](https://huggingface.co/jobs/devjuneja789/6af94f0d23985fb9cd82f91d)
completed; the training and exact-context results are recorded below.

## Training result

HF Job `6af94f0d23985fb9cd82f91d` completed. Run `20261115T062137Z`
applied all 60 updates, four epochs, zero skips. The joint dummy-logit check,
all-15-batch finite-gradient preflight, mask/weight audits and save/reload pass.
Every completion token, including both native closing tokens, has positive
effective loss. Source references/targets/reviews remain unchanged.

Training runtime 208.4633 seconds; peak allocated CUDA memory 8,231,636,480
bytes on T4. Epoch mean joint loss: 1.89387, 1.34156, 1.00691, 0.86637.
These are within-objective learning statistics, not task-success evidence or
directly comparable values to the previous objectives' losses.

Private artifact `devjuneja789/terminal-slm-tiny-aya-joint-60-v1`, pinned revision
`36f7fecf593e523ca18b910d635465533a9a0e41`, subfolder
`20261115T062137Z/adapter`. Local results under
`research/results/tiny-aya-joint-sft/20261115T062137Z` include full result,
training history, provenance and independent training gate. The initial reload
probe still chooses shell on a read target; no positive action claim follows.

Exact-context Job
[6af950c929dd05ff03684d11](https://huggingface.co/jobs/devjuneja789/6af950c929dd05ff03684d11)
completed, run `20261115T063016Z`. All 45 context/prediction records were
recovered from the original log and passed independent analysis.

## Exact-context results and comparison

| Metric | Tool-only EXP-022 | Joint objective |
|---|---:|---:|
| Schema-valid outputs | 14/15 | 14/15 |
| Valid correct tool choices | 8/15 | 8/15 |
| Correct read_file choices | 0/2 | 0/2 |
| Correct write_file choices | 2/3 | 2/3 |
| Correct finish choices | 0/4 | 0/4 |
| Exact actions | 3/15 | 3/15 |
| Teacher-forced full-action token accuracy | 87.53% | 87.53% |
| Teacher-forced full-action NLL | 0.78506 | 0.75404 |
| Generated tokens | 889 | 889 |
| Cap hits | 0 | 0 |

**Primary criterion not met.** Exact-action nonregression meets its minimum
3/15, but the 15/15 schema goal also fails. All 15 adapter greedy token
sequences and teacher top-1 token arrays are identical to EXP-022, despite the
modest change in probabilities/NLL. All 15 base controls likewise reproduce.
All 30 generations use the same native stops and end on END_RESPONSE; all
context tokens, retained labels and new effective weights reproduce exactly.

A read-only Hub LFS hash check confirms the model files differ: old adapter
SHA-256 `bce0b7f795f4d48ba1df4cd572afd1901581949dff78c43fec1b0c280d55fb05`,
new `8e3adeb1011c708b1f2ada5608af7a5fdfc16076de5759b72df437a10e5150cf`.
Metadata and the inference loader explicitly pin the new repository/revision.
This is a changed, reloaded adapter whose confidence changes did not cross
greedy decision boundaries, not evidence of accidentally reusing the old file.

Static output findings are unchanged: B021 writes default timeout 1 instead
of fixing page_size; B026 attempts malformed write JSON and incomplete source;
B041 changes the intended default and retains both functions but removes the
header/import and is not the approved full file. It is not runtime-validated.
The three exact actions are primary named tests, upper-bound checks repeat
those tests, complete-read targets select shell, and all four finish targets
select shell. No proposed command, file change or validator is executed here.

## Conclusion and next decision

At this fixed 60-update budget, the joint first-token/completion regimen does
not improve greedy action behavior over tool-only training. Positive argument
supervision modestly improves full-action likelihood but does not repair the
observed malformed or incorrect edits. It does not establish a model-capacity
limit or prove that a longer fit would succeed. Avoid attributing individual
effects separately to first-token emphasis versus restored completion loss;
they form this one prespecified joint treatment.

Stop after the fixed experiment. Do not start an agent replay, held-out run,
new data admission or more training automatically. A future bounded fit-budget
study is more informative than declaring success from loss or cycling through
more objective variants, but would need its own fixed protocol.

## Artifacts and verification

Under `research/results/tiny-aya-joint-contexts/20261115T063016Z`: context and
prediction JSONL, metadata, summary, independent analysis, comparison, read-only
adapter hash check, readable summary, provenance and job-inspection records.
Training artifacts include all 60 history entries and a positive-coefficient/
gradient/update/reload gate. Original receipts, reviews and admitted data are
unchanged. Local verification: 153 tests run (150 pass, three optional Torch
checks skipped), plus actual CUDA objective and all-batch gradient checks.
Shell syntax, compilation and `git diff --check` pass. User-authorized commands
and cloud submissions completed; no additional user information was needed.
