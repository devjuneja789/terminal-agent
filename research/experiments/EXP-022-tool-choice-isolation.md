# EXP-022: Isolate tool-name and finish learning

## Fixed plan before execution

EXP-021 found that the corrective adapter still chooses no write_file or finish
on the exact reviewed contexts. Test whether isolating action-name gradients
improves these decisions, rather than assuming runtime coverage is the cause.

Control: EXP-020 training + EXP-021 exact-context predictions. Keep exactly the
same four admitted demonstrations and 15 immutable targets, source row hash
`03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b`.
Start fresh from the same pinned Tiny Aya base, with the same rank-8 q_proj/v_proj
LoRA, NF4/FP16 compute, FP32 trainable adapters, seed 42, batch/accumulation 1,
learning rate 1e-4, linear schedule, no warmup, clipping 1, AMP scale 8192,
maximum length 4096, and exactly 60 updates/four epochs. No old-data merge,
new data, initialization change, extended budget or prompt/protocol changes.

**Only the optimized objective changes:** mean causal cross-entropy over the
native JSON `tool_name` value tokens, including `finish`. Each selected token
has coefficient 1; every prompt token and all other completion tokens have
coefficient 0. Full reviewed text and full label/template checks remain intact;
retained labels do not imply nonzero optimization weight. Arguments, file text,
JSON closure, END_RESPONSE and turn EOS receive no training loss in this run.
The previous loss-weighting profile remains only for locating/auditing spans;
its 8x/8x/4x coefficients do not contribute to this objective. No custom syntax
or synthetic classification labels are introduced.

Before optimizer updates, verify all 15 label boundaries, effective binary
weights and tool spans. A CUDA dummy-logit autograd check must prove zero
gradient at nonselected positions and invariance of the choice loss to their
logits. It touches no model parameters. Abort on a failed gate, truncation,
nonfinite training loss, skipped update, wrong objective or unchanged adapter.
Store audits, full training history and a reloaded adapter in a new private HF
repository. Single successful T4-small training Job, 25-minute cap; no adaptive
budget, learning-rate or data changes. Failed numerical startups remain recorded.

Then run the unchanged exact-reviewed-context procedure: fresh base plus the
new adapter, 15 contexts each, greedy seed 42, NF4/FP16/eager, 256 new-token cap,
native stops `[3,6,261001]`, no trimming, tools, training or held-out tasks.
Reproduce actual new training tokens/labels/effective weights before inference.
Require 15 context audits and 30 predictions, independent recomputation, and
compare against archived EXP-021 outputs. Repeat the base arm as a control.
One T4-small inference Job, 30-minute cap, 20-minute between-case budget.

Primary learning criterion: **15/15 strict-schema-valid correct tool choices**
on the known contexts, explicitly including all three write and four finish
targets. Also report per-tool counts, exact argument/action matches, schema
validity, teacher-forced scores, cap/closing status and qualitative changes.
This is a memorization diagnostic, not generalization or successful repairs.
Even passing the criterion does not demonstrate arguments/file preservation;
those tokens were deliberately not optimized. Stop after the fixed 60 updates
and complete diagnostic regardless of outcome. No automatic full-agent replay,
further training, data admission or held-out access.

## Status

Local verification: 150 tests run, 147 pass and three optional Torch checks
skip. Four new tests verify span-only binary weights, unmodified source/target
staging, objective-aware artifact validation and unchanged context procedure.
Shell syntax, compilation and `git diff --check` pass. Training Job
[6af9466623985fb9cd82f08b](https://huggingface.co/jobs/devjuneja789/6af9466623985fb9cd82f08b)
is the recorded failed numerical startup; the corrected training and completed
diagnostic results appear below.

### Recorded numerical startup failure and versioned correction

The first Job `6af9466623985fb9cd82f08b` aborts at attempted step 1 because FP16
AMP skips the update at scale 8192. No optimizer update is applied and no trained
artifact/result is produced. Keep its raw log and preserve the original config
as `configs/tiny_aya_tool_choice_sft_v1.json`. The objective's dummy-logit
gradient-isolation check was reached before training; actual optimizer safety
checks prevented silently treating an overflow as a completed update.

Revise the numerical startup profile to
`terminal-slm-tool-choice-only-60-v2-scale256`: fixed AMP initial scale 256,
plus a real all-15-batch scaled-gradient preflight before optimizer creation.
This probe uses the same checkpointing/autocast configuration, clears gradients,
performs no optimizer updates and aborts if any gradient/loss is nonfinite or
the total gradient norm is zero. It does not search scales or tune the objective.
The objective, initialization, examples, LR/schedule and 60-update budget remain
fixed. This is an explicitly recorded numerical correction, not a completed
same-scale comparison with EXP-020. One corrected launch follows local checks;
no automatic further numerical retries or training extension are planned.

Static control audit (`research/results/tiny-aya-tool-choice-sft/objective-preflight.json`)
finds tool-name coefficients account for 5.6–8.8% of normalized control loss
weights on write examples and 17.6–23.2% on finish examples. This is loss
coefficient accounting, not measured gradient attribution or proof of causality.

## Corrected training result

Job [6af947e823985fb9cd82f1c2](https://huggingface.co/jobs/devjuneja789/6af947e823985fb9cd82f1c2),
run `20261115T055053Z`, completes 60 optimizer updates/four epochs with zero
skips. All 15 effective binary-weight audits, retained label/template checks,
dummy-logit isolation and all-batch real-gradient preflight checks pass. Maximum
preflight unscaled gradient norm 14.5877 at scale 256; no optimizer is created
and all gradients are cleared during that preflight. Arguments and both native
closing tokens retain zero effective supervision in this experiment.

Training runtime 209.2321 seconds, peak allocated CUDA memory 8,231,636,480 bytes
on Tesla T4. All 1,916,928 trainable adapter parameters use FP32, with the frozen
NF4 base computing in FP16. Adapter weights change and reload successfully.

| Epoch | Mean tool-name loss |
|---|---:|
| 1 | 1.40843 |
| 2 | 0.91612 |
| 3 | 0.63937 |
| 4 | 0.51773 |

Overall mean tool-name training loss 0.87041. These values are not directly
comparable to EXP-020's full-completion weighted loss. The reload probe still
chooses shell on the first read target, so the complete context evaluation is
required; declining training loss alone is insufficient.

Private artifacts: `devjuneja789/terminal-slm-tiny-aya-tool-choice-60-v1`,
revision `2bb02aafefcfcb5de5a4c79737c8cbb20dd6a0d9`, subfolder
`20261115T055053Z/adapter`. The empty private destination left by the failed
startup was reused; no previous trained weights were overwritten. Local
`research/results/tiny-aya-tool-choice-sft/20261115T055053Z` includes the complete
`.result.json`, `.training-history.jsonl`, `.provenance.json` and independent
`.training-gate.json`. The failed-startup record remains separate.

Exact-context evaluation Job
[6af9499b29dd05ff03684542](https://huggingface.co/jobs/devjuneja789/6af9499b29dd05ff03684542)
completed, run `20261115T055808Z`. No action-quality result is inferred from
training loss alone.

## Exact-context results

| Metric | EXP-021 corrective adapter | Tool-choice-only adapter |
|---|---:|---:|
| Strict-schema-valid predictions | 15/15 | 14/15 |
| Valid correct tool choice | 6/15 | 8/15 |
| Exact parsed action | 3/15 | 3/15 |
| Correct read_file choice | 0/2 | 0/2 |
| Correct write_file choice | 0/3 | 2/3 |
| Correct shell choice | 6/6 | 6/6 |
| Correct finish choice | 0/4 | 0/4 |
| Teacher-forced exact full sequence | 3/15 | 0/15 |
| Mean teacher-forced full-action token accuracy | 89.64% | 87.53% |
| Mean teacher-forced full-action NLL | 0.52794 | 0.78506 |
| Generated tokens | 618 | 889 |
| Cap hits | 0 | 0 |

**Primary 15/15 tool-choice memorization criterion: not met.** All three raw
write-target outputs attempt write_file, but the malformed B026 action is not
counted as a valid choice. Tool-name spans are top-1 exact under teacher forcing
on 9/15 targets (six shell plus three write), versus 6/15 previously; read and
finish spans remain incorrect. The only exact actions are still the primary
named tests at indices 02/06/11. Upper-bound regression checks repeat those
primary tests instead of the reviewed command.

All 15 repeated base controls reproduce the exact EXP-021 generation tokens,
decoded outputs and teacher top-1 token arrays. All 15 exact context/token/label/
effective-weight checks and 30 native-stop checks pass, and all 45 records are
recovered from the original log. Every generation ends on END_RESPONSE without
cap hits. The independent analyzer checks the unchanged inference configuration,
new adapter lineage, changed training fields, actions, metrics and control parity.

## Factual/static review of proposed edits and finishing

- B021 now selects a valid write_file, retains both functions, but changes
  DEFAULT_TIMEOUT_SECONDS to 1 and leaves `page_size()` returning
  `min(value, MAX_PAGE_SIZE)`. It does not implement the requested lower bound.
- B026 attempts write_file but omits a JSON closing brace. Its proposed text
  also drops page_size/imports and changes the default to 25. Strict validation
  rejects the action; do not repair its JSON or credit it as an executable write.
- B041 produces valid JSON with the intended default 15 and both original
  functions. It also removes the module header/import and is not byte-equal to
  the reviewed full file. This is a plausible conditional default update with
  static function preservation, not a newly executed or validated repair.
- Both complete-read targets still produce shell tests instead of read_file.
  All four finish targets still produce shell commands, including an unnecessary
  upper-bound test or grep. There are no final-answer predictions to credit as
  legitimate paraphrases or successful termination.

No generated command, write or validator executes in this diagnostic. Do not
turn these static observations into benchmark results. The loss deliberately
excluded argument/code/closure supervision, and degradation in those secondary
metrics is retained rather than hidden.

## Conclusion and next decision

This regimen can shift write-name selection on the reviewed contexts, but it
does **not** learn the full read/write/finish transitions or improve exact
actions. Ending and valid edit arguments remain blockers. The separate AMP
startup correction means this is not a perfectly matched numerical control;
do not attribute every difference exclusively to the loss objective. No model
capacity limit, successful repair, or held-out improvement is established.

Stop here: no full-agent replay, longer run, data expansion or output admission.
The next research decision must address the unchanged finish/read choices and
retain argument/closure supervision before treating this objective as an agent
improvement. Another broad replay is not justified by 8/15 choices and zero
finish predictions. Any further targeted experiment requires its own fixed
plan; none follows automatically. Nothing further is required from the owner
for this completed diagnostic.

## Artifacts

Training artifacts are listed above. Under
`research/results/tiny-aya-tool-choice-contexts/20261115T055808Z`:
`.context.jsonl`, `.prediction.jsonl`, `.metadata.json`, `.summary.json`,
`.analysis.json`, `.isolation.json`, `.md`, `.provenance.json` and
`.job-inspection.txt`. Provenance hashes the raw log, training/analysis sources,
independent audits and job status. All three job outcomes, including the failed
startup, are retained. Owner receipts/reviews/admitted exports are unchanged.
