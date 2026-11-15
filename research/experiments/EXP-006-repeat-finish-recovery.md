# EXP-006 — Repeat-action finish recovery

**Status:** Implementation and local tests complete; cloud inference not run.
**Date:** 2026-11-12
**Split:** `train-support` only

## Hypothesis

When Tiny Aya repeats a valid tool call after receiving useful evidence, one
additional model turn that requires a `finish` action may recover an answer
without rerunning the tool. This tests the observed loop-control failure; it is
not a model-training experiment and does not establish coding-task reliability.

## Change

The support-only `support-v5-repeat-finish-retry` policy detects a duplicate
action, does not dispatch it, and asks the model once to answer from prior tool
observations with tool calls disabled. Only a non-empty model `finish` action
is accepted. If the retry is malformed or proposes another tool, the harness
stops with an empty answer and marks the output as harness control flow. No
harness-authored answer is used as a model target. The baseline and v4 policy
remain unchanged.

The evaluator maps each returned AgentTune action to its source model output so
the retry and the blocked duplicate remain correctly ordered in the trajectory
and token/turn metrics.

## Focused support check

The preselected support subset is `B016`, `B046`, and `B041`:

- B016 and B046 test whether the model can finish using already observed
  evidence after repeating an action.
- B041 checks whether an edit task results in an actual `write_file` call and
  validated workspace change.

The bundle builder verified that the config contains only these support IDs,
the split manifest exposes only `train-support`, and no held-out task file or
baseline config is included.

## Local verification

- `tests.test_support_policy`: 2 passed.
- Full local suite: 67 passed.
- Python compilation, shell syntax checks, and `git diff --check`: passed.

## Cloud result (2026-11-12)

The focused T4 run completed as `20261112T200940Z`. It accessed only
`train-support` (`heldout_accessed: false`). The raw job summary reports 2/3
benchmark contract passes and 2 automated successes, with zero
training-eligible candidates. Preserve the run's exact configuration and
artifact hashes:

- Config: `terminal-slm-tiny-aya-support-rollout-v5`, SHA-256
  `1534850daa16a18f6848360b289a4199cbec0f168c37072f2a2af7e01aafa081`.
- Model: `CohereLabs/tiny-aya-global` revision
  `af89d219b53ed9b13b8a4645f9c8028973510324`, float16.
- Prompt SHA-256:
  `dea72255131712f5c0a86b99aef6513707ec3311fcd4a6d736e60bf8c66460a1`.
- Support taskset SHA-256:
  `b224030602c9299d1b5f1133ea300a0fd624e50a8a6b9a7fb070f883dd69a537`.
- Runtime: Python 3.12.16, PyTorch 2.11.0+cu130, Transformers 5.22.1,
  AgentTune 1.1.3; Tesla T4, 15 GiB.
- Artifacts: `research/results/tiny-aya-support-rollouts/20261112T200940Z.jsonl`,
  `.metadata.json`, `.summary.json`; cloud log
  `research/results/tiny-aya-support-rollouts/20261112T200516Z.log`.

### Trace audit

- **B016:** the model used grep and read_file, then answered with the declared
  variable name. It read only line 8, not the `request_timeout()` body, and did
  not answer whether the function reads the variable. The automatic contract
  passed because it only checked the variable name and tool presence. This is
  an under-specified success contract, not verified task completion.
- **B046:** the model repeated the test command; the duplicate was blocked and
  the finish-only retry generated a final answer. The answer correctly cites
  the observed `10 != 25` mismatch, but says the `request_timeout` environment
  variable is set to 10. The snapshot shows instead that `request_timeout()`
  returns the constant `DEFAULT_TIMEOUT_SECONDS` and never reads
  `REQUEST_TIMEOUT_SECONDS`. Its factual review remains necessary; the keyword
  contract pass is not proof of a correct explanation.
- **B041:** the model's proposed `sed` command printed transformed file content
  to stdout; it did not edit the file. No `write_file` call occurred, the
  validator still failed, and the workspace remained unchanged. The finish-only
  retry reported the observed failure rather than completing the requested
  edit.

The retry mechanism therefore recovered a model-generated finish after
repetition on B046 and B041, but the pilot produced **zero verified code edits**
and no training data. The raw summary's 2/3 is a contract metric; it must not be
reported as 2/3 factually completed tasks. All three records require review,
all remain training-ineligible, and no SFT/GRPO run should use them. B016's
contract should require reading the function body and reporting the
environment-variable behavior before any future support run.

## Decision

Keep the repeat-finish recovery as a bounded harness mitigation, not evidence
that Tiny Aya learned to stop. Do not spend credits on another identical
rollout or begin training from the rejected candidates. The unresolved central
limitation is coding behavior: the model still did not call `write_file` on
B041. If continuing the hackathon submission, present this as a narrow
reliability diagnosis and only attempt post-training after creating
human-authored, validator-checked support demonstrations with clear provenance.
