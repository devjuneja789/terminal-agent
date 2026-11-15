# Final trace review

Inspection of the actual final records by the implementation assistant, not a
human/owner approval receipt or new training admission. Run `20261115T071710Z`; immutable
candidate `1eed40612640c78d88d9fe0cac36e06b1148d1fd`. The benchmark remains
unchanged. This review supplements its deterministic scores.

## Nominal passes

| Arm / task | Frozen contract | Trace evidence and limitation |
|---|---|---|
| Base / B010 | PASS | `grep` observes `shop/users.py:4:def display_name(...)`. The reported location is correct. Its generic dictionary-to-name purpose is consistent with the implementation, but the body and fallback logic were not inspected. |
| Adapter / B010 | PASS | Same correct location. A second grep searches the path as a text pattern and finds nothing. The answer explicitly guesses the purpose from the signature; the actual implementation prefers `display_name`, falls back to `username`, then empty string. This is partial evidence, not a fully grounded implementation explanation. |
| Adapter / B050 | PASS | Actually runs `python -m unittest`, obtains exit 1 and the real `10 != 15` default-timeout failure among six failures. Its final answer mixes unrelated tests, incorrectly says the environment test asserts 10, and says the lower-bound test fails while both actual/expected are 1. The observed lower-bound mismatch is `0 != 1`. It does not clearly explain the requested default mismatch. The contract checks only `fail`, shell presence and expected validator exit 1; it cannot detect these factual errors. |

**The nominal 10% → 20% change does not establish an additional factually
reliable task completion.** Do not present the B050 pass as a successful grounded
explanation. AuditKIT's verified-contract count has the same rubric limitation;
it is not an independent semantic judge.

## Remaining failures

- B005: both search unsuccessfully or provide an incorrect/unknown package.
- B015/B020: both miss the declared environment setting or fail to inspect the
  implementation. Adapter often stops with `Unknown` after an unhelpful grep.
- B025/B030/B035: no file edit; the reserved-slug and boolean-validation
  repairs remain unimplemented. Adapter B030 does not run the requested test.
- B040: adapter calls write_file twice with path `.`. Both calls are rejected
  because the path is a directory. Proposed snippets concern empty-directory
  checks, not adding the readiness route. No file changes.
- B045: adapter emits two actions in one JSON array. The one-action policy
  rejects it before execution. The proposed content also targets `.` and a
  timeout of 30 instead of the required 15. No file changes.

Across all twenty final episodes, no task file changes are recorded. Both
models therefore remain **0/5 on the code-repair/update subset**. Some final
responses claim changes or success without edits; the executable validators
correctly fail them. Nonzero expected test evidence is preserved separately
from genuine executor errors.

## Interpretation

Longer fixed-budget training resolves the reviewed-context tool-choice/finish
learning failure: 15/15 correct choices, two exact complete writes and no
schema errors. These conditional gains fail to transfer to reliable initial
searches, edit paths and complete agent repairs. The four corrective examples
do not cover the variety of initial states and arguments in the evaluation.
This is evidence of limited transfer with this recipe, not proof of an
intrinsic model-capacity limit.

The weak factual rubric was already a documented limitation. Preserve its
actual v1 scores and the semantic counterexample instead of silently changing
success labels. A future stricter benchmark needs its own version and a fresh
matched baseline. No held-out-derived demonstration, reward, prompt adjustment
or further training was created from this inspection.

## Subsequent support-only experiment

EXP-027 was conducted after this review using support-only approved targets.
It is a separately versioned exploratory run, not a change to these held-out
records or the frozen adapter above. Its task-start adapter passed two of four
known support tasks; the two successful edits passed independent full-file
preservation and executable regression checks. It failed the other two tasks
and has no held-out evaluation. See the dedicated [EXP-027 report](../research/experiments/EXP-027-task-start-curriculum-plan.md).
