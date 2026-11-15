# EXP-021: Predictions on the exact reviewed corrective contexts

## Fixed plan before execution

Distinguish conditional action learning from failure to reach corrected states
in EXP-020's full replay. No new training, optimizer, backward, tools, benchmark
tasks, held-out access, new data admission or prompt rewriting. Stage only the
four admitted references, their exact 15 target rows and completed training audit.

Compare fresh Tiny Aya base (revision
`af89d219b53ed9b13b8a4645f9c8028973510324`) and the EXP-020 corrective adapter
(revision `3c151e4280bab3fe27c25a53992a3cc17fc785fc`, repository
`devjuneja789/terminal-slm-tiny-aya-corrective-60-v1`, subfolder
`20261114T222003Z/adapter`). Verify tokenizer equivalence. Use the same pinned
runtime, NF4 double quantization/FP16/eager attention, seed 42, greedy generation,
cache, 256-token cap and native stops `[3,6,261001]` in both arms. No context
trimming. One HF T4 Job, 30-minute cap, 20-minute between-case inference budget.

Before loading a model, reproduce every actual training prompt/target token,
label position, weight and masked/target count using the exact saved annotations.
No trainer is constructed; labels are reconstructed from a verified chat prefix
and checked against the actual training record, not assumed equivalent.

For each of 15 contexts and both arms, record:

- Raw greedy completion/token IDs, strict one-action schema validity, tool-name
  match and parsed action equality (JSON whitespace/key order do not matter).
- For write actions, exact path/content equality and static syntax/function
  inspection; never execute generated code or validate it as a completed repair.
- Teacher-forced next-token accuracy/NLL, exact teacher sequence, critical
  tool-name/changed-code/JSON-closure token scopes and first token divergence.
- Exact context, reviewed gold action, token/label audit, cap/closing behavior,
  timing and immutable lineage. Require all 15 context audits + 30 predictions.

Exact finish text is stricter than factual correctness: inspect all four final
answers and distinguish legitimate paraphrases from unsupported claims. No
tool/result observations are generated in this experiment. Teacher-forced
scores use gold preceding tokens and cannot substitute for correct greedy
actions or full agent success. If the adapter still fails the reviewed actions,
missing runtime coverage alone cannot explain EXP-020; if it succeeds here,
coverage remains a hypothesis, not a proven causal explanation.

Stop after the complete diagnostic, independent aggregation and factual review.
Do not automatically train longer, merge data, change benchmark contracts or
run held-out tasks.

## Status

Complete. Local checks pass: 146 tests run (143 pass, three optional Torch checks skipped),
including four new context-diagnostic tests. Shell syntax, compilation and
`git diff --check` pass. GPU Job
[6af8e42629dd05ff0367e923](https://huggingface.co/jobs/devjuneja789/6af8e42629dd05ff0367e923)
independently confirmed COMPLETED, 276 running seconds. Run `20261114T224529Z`,
configuration SHA-256
`ef50a3377ebfc1b2e1bba99b7e411ee3c07a8310c5583e8415f37c65b8c07ba9`.
Tesla T4; all package, quantization and hardware fields are in raw metadata.
At the displayed T4-small rate $0.40/hour, running duration corresponds to
approximately $0.031, not an account billing receipt.

## Results

| Metric | Base | Corrective adapter |
|---|---:|---:|
| Exact training contexts reproduced | 15/15 | Same 15 contexts |
| Schema-valid greedy outputs | 15/15 | 15/15 |
| Correct tool choice | 1/15 | 6/15 |
| Exact parsed action | 1/15 | 3/15 |
| Teacher-forced exact target sequence | 0/15 | 3/15 |
| Mean action token accuracy | 86.69% | 89.64% |
| Mean action NLL | 0.81057 | 0.52794 |
| Teacher-forced tool-name span exact | 1/15 | 6/15 |
| Teacher-forced changed-code span exact | 1/3 | 1/3 |
| Teacher-forced JSON-closure span exact | 11/15 | 14/15 |
| Generated tokens | 529 | 618 |
| Token-cap hits | 0 | 0 |

All 30 generations use `[3,6,261001]` and stop once on END_RESPONSE. Exact
prompt, training-target token IDs, label positions/counts and weights reproduce
for all 15 actions; no context trimming occurs. Independent analysis recomputes
each strict JSON comparison, teacher-token/scoped accuracy, aggregates and
termination checks from raw records. All 45 records recover from 114 bounded
chunks in the original live log; no recovery job or inference retry is needed.

## Action-level diagnosis

| Expected action | Targets | Base tool choice / exact action | Adapter tool choice / exact action |
|---|---:|---:|---:|
| read_file | 2 | 0 / 0 | 0 / 0 |
| write_file | 3 | 0 / 0 | 0 / 0 |
| shell | 6 | 1 / 1 | 6 / 3 |
| finish | 4 | 0 / 0 | 0 / 0 |

The adapter's only three exact predictions are the primary named tests for
B021, B026 and B041 (indices 02, 06, 11). At the three upper-bound regression
checks (03, 07, 12), it instead repeats each task's primary named test. Thus
tool-name matches on all six shell targets do not imply correct arguments.

At B021/B041 write targets (01, 10), both models choose read_file again, despite
the correct complete source already being supplied. At B026's write target
(05), the adapter proposes `git config -f shop/config.py REQUEST_TIMEOUT_SECONDS`
through shell; it does not produce an edit. At the two complete-read targets
(00, 09), it proposes the named test instead of reading.

At all four finish targets (04, 08, 13, 14), the adapter chooses another shell
command rather than finish. The first three propose the upper-bound test;
B046 proposes a grep for REQUEST_TIMEOUT_SECONDS. Consequently there are no
generated final answers on the four reviewed finish contexts, and no finish
paraphrases to mark correct or incorrect. Likewise there are no generated write
actions to assess as preserved full files. No commands execute here.

The adapter emits 13 shell and two read_file actions in total; the base emits
eight shell, two read_file and five misplaced finish actions. Schema compliance
is perfect in these supplied contexts, so malformed JSON is not the immediate
limitation on these 15 cases. Tool selection, command selection, editing and
termination remain unreliable. Improved teacher-forced closure scores and NLL
do not establish learned repair behavior; changed-code top-1 span exactness
does not improve and uses gold action tokens as preceding context.

## Conclusion and next step

**Conditional action learning remains insufficient. Failure to reach corrected
states cannot be the sole explanation of EXP-020.** Supplying the exact reviewed
histories still yields no correct write or finish choices. Some test-running
and token-prediction behavior improves, but the model has not learned the
required action transitions reliably. This does not establish a fundamental
model-capacity limit, prove that additional epochs would solve it, or identify
one causal optimization defect.

Do not spend on a larger end-to-end replay or broaden the dataset on a
coverage-only assumption. The next proposed experiment should isolate learning
the critical tool choice and finish transitions on these same approved targets,
with a fixed stopping criterion and exact-context action scores, before another
full-agent run. Any changed loss, schedule or initialization requires a separately
versioned plan; none is executed here. No held-out tasks, new training targets,
new training or terminal tools were used.

## Artifacts

Under `research/results/tiny-aya-corrective-contexts/20261114T224529Z`:
`.context.jsonl` (15 exact contexts/labels/weights), `.prediction.jsonl` (30 raw
greedy outputs/IDs and teacher-forced token predictions), `.metadata.json`,
`.summary.json`, `.analysis.json`, `.md`, `.provenance.json` and
`.job-inspection.txt`. Provenance records log, analysis, source and job-status
hashes. The raw console log is retained locally and gitignored. Owner receipts,
review sidecars, admitted exports and the previous training artifacts remain
unchanged; no predictions are automatically admitted for training.
