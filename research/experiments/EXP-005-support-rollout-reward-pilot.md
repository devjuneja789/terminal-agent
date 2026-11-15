# EXP-005 — Support-only rollout and deterministic reward pilot

**Status:** Pilot reviewed and rescored with corrected contracts and scorer.
**Phase:** 4
**Purpose:** Generate a small set of support-only agent trajectories, validate
the reward contract, and identify candidates for manual review before any
training-data conversion.

## Scope and split protection

The pilot selection is fixed in
`training/agenttune/prepare_hf_support_bundle.py` as one support task per
category: `B001, B006, B011, B016, B021, B026, B031, B036, B041, B046`.
The builder checks that every selected task belongs to `train-support` and
that the ten tasks cover ten distinct categories.

The job bundle includes only `train_support.jsonl`, a support-only manifest,
the common miniapp snapshot, required tool/runner code, and a generated
support config. It excludes `heldout_eval.jsonl` and
`tiny_aya_eval.json` (which contains held-out IDs). At runtime, the loader
requires a manifest with exactly the `train-support` split and refuses to run
if a held-out task file is present. The generated config inherits the frozen
baseline model revision, prompt, tool schemas, decoding, and package pins, but
contains only the support pilot IDs.

## Reward and review rules

The deterministic score is defined in
`training/agenttune/support_rewards.py`:

- Validator-backed code-change categories must fail validation on the pristine
  snapshot, produce an actual task-file change, and pass the benchmark's final
  response/tool/validator contract after the agent run. Calling a required
  tool or writing a no-op cannot receive task-success reward.
- Base task-success reward is `1.0` only when those conditions hold.
- Small fixed penalties apply to malformed/schema-invalid actions, failed tool
  executions, repeated identical calls, and calls beyond the four-call free
  allowance. The combined penalty is bounded below at `-0.25`.
- Repository navigation, function/config/environment discovery, and test
  explanation use contracts that cannot fully verify factual correctness.
  These candidates receive `reward: null` until a human checks them against
  their support snapshot.
- Every record remains `training_eligible: false` and human review pending.
  No rollout is silently converted into SFT or GRPO data.

The evaluator records pre-agent validation, changed file paths, post-agent
validation, complete prompt/model/tool trajectories, reward components, and
task provenance. A runtime reward-integrity guard rejects positive reward if
the validator-backed task-success check failed.

## Configuration and execution

- Base model/config: Tiny Aya Global at the same pinned revision and decoding
  configuration as EXP-004.
- Runtime: the pinned cloud Python/PyTorch/Transformers/Accelerate/AgentTune
  versions, on a Hugging Face T4 job.
- Workspace: a fresh snapshot copy per task; standard Terminal-SLM workspace
  tools and bounded time/output settings.
- Job timeout: 40 minutes. The launcher prints live `t4-small` pricing before
  submitting; review the current credit balance first.
- Launch command: `./scripts/run_tiny_aya_support_rollouts.sh`.
- Output: chunked records, environment/config metadata, and summary under
  `research/results/tiny-aya-support-rollouts/`.

## Pilot run (2026-11-09)

- Run: `20261109T070739Z`; config: `terminal-slm-tiny-aya-support-rollout-v1`.
- Split: `train-support`; IDs: `B001, B006, B011, B016, B021, B026, B031,
  B036, B041, B046`. The job reports `heldout_accessed: false`.
- Runtime: 124.25 seconds; 10 candidate trajectories and reward audits.
- Benchmark contracts passed: 2/10 (`B006`, `B046`), both factual-answer
  categories. They remain pending human factual review.
- No code-change candidate passed the automated task-success reward gate.
  The reviewer found that the five `rejected_no_op_or_baseline_validator_passes`
  cases were all caused by no file change, not pristine validators passing.
  Two tasks failed validation (`B026`, `B041`).
- Six candidates had invalid tool calls. The summary reports five scored
  candidates, mean non-null reward `-0.03`, all 10 requiring human review, and
  zero training-eligible candidates.

Artifacts:

- `research/results/tiny-aya-support-rollouts/20261109T070739Z.jsonl`
- `research/results/tiny-aya-support-rollouts/20261109T070739Z.summary.json`
- `research/results/tiny-aya-support-rollouts/20261109T070739Z.metadata.json`
- Cloud log: `research/results/tiny-aya-support-rollouts/20261109T070550Z.log`

These are rollout outcomes, not a training result. No weights were updated or
SFT/GRPO data produced.

## Corrected reviewed rescore (2026-11-10)

The raw trajectory file was preserved. The scorer reprocessed it against the
strengthened support contracts and applied the reviewer sidecar into
`research/results/tiny-aya-support-rollouts/20261109T070739Z.reviewed.jsonl`.
The resulting summary is
`research/results/tiny-aya-support-rollouts/20261109T070739Z.reviewed.summary.json`:

- Contract passes: 0/10 (the original B006 pass omitted `read_file`; B046
  never ran the required test or observed its `10 != 25` assertion).
- Human review disposition: rejected for all 10 candidates.
- Non-null scores: 3; average `-0.0367`.
- Training-eligible candidates: 0.
- The sidecar and corrected scores disagree for some records; those records
  remain unrewarded and are flagged in each row's `review_audit`.

This is a rescore of saved evidence and human review, not new model inference.

## Support v2 outcome and v3 behavior iteration

The v2 support-only run `20261110T132907Z` completed in 145.05 seconds on the
same ten support IDs and reports `heldout_accessed: false`. It had 0/10
contract passes, 6 invalid-tool-call failures, 2 incomplete runs, and 2 task
contract failures. Five factual candidates remain pending human review; five
change candidates were rejected for no file change. There are zero
training-eligible candidates. Artifacts are:

- `research/results/tiny-aya-support-rollouts/20261110T132907Z.jsonl`
- `research/results/tiny-aya-support-rollouts/20261110T132907Z.summary.json`
- `research/results/tiny-aya-support-rollouts/20261110T132907Z.metadata.json`
- Cloud log: `research/results/tiny-aya-support-rollouts/20261110T132741Z.log`

Trace inspection found `grep` patterns aimed at filenames rather than file
contents (B001), failure to search/inspect the exact configuration symbol
(B011), repeated empty grep or repeated read calls (B001/B016), premature
answers without required `read_file` evidence (B006/B011), malformed shell
`parameters` arrays or unsupported `args` fields (B021/B026/B031/B036/B041/B046),
and shell timeouts set to 0.001 seconds. Tool observations were delivered to
the model as Python dictionary representations, which is a context-formatting
weakness. These outcomes show that the v2 prompt change alone did not improve
task success; they do not support starting training.

## v3 support rollout (2026-11-10)

Run `20261110T140104Z` used config
`terminal-slm-tiny-aya-support-rollout-v3` on the ten support IDs and reports
`heldout_accessed: false`. It completed in 198.89 seconds with 0/10 contract
passes, 0 automated task successes, 0 training-eligible records, and mean
non-null reward `-0.106`. Failure categories: one incomplete run, three task
contract failures, one tool execution failure, and five validation failures.
There were no invalid-tool-call failures (v2 had six), so JSON observation
formatting and the stricter argument prompt improved schema compliance. The
agent still repeated shell commands, skipped required inspection tools, failed
to edit all five change-task workspaces, and hallucinated a path in B001. Five
factual tasks remain pending human review; five change tasks were rejected for
no file change.

Artifacts:

- `research/results/tiny-aya-support-rollouts/20261110T140104Z.jsonl`
- `research/results/tiny-aya-support-rollouts/20261110T140104Z.summary.json`
- `research/results/tiny-aya-support-rollouts/20261110T140104Z.metadata.json`
- Cloud log: `research/results/tiny-aya-support-rollouts/20261110T135938Z.log`

## Next support-only behavior iteration

The next candidate config is `terminal-slm-tiny-aya-support-rollout-v4`.
It retains JSON observations and stops any repeated identical tool call unless
a valid `write_file` action intervened, allowing a test rerun after an edit
while stopping unchanged shell loops. It also gives the selected support tasks
more actionable source paths and exact test commands, based on observed
navigation failures. The model revision, decoding settings, tool schemas, and
support IDs remain the same. Held-out task records and the frozen baseline
config remain unchanged. The bundle was prepared at
`/tmp/terminal-slm-support-v4-ready-20261110`, contains no held-out task file,
and uses support task revision `2026-11-10-review-clarified-r4` with taskset
SHA-256 `1294a4cd07bb3d27f759f30c572b944993e08cf58e64554929622779732c1035`.
Prompt SHA-256 is
`6cd34d2dfa7310dfd4211c487ffe0a4240519c3890a7754308d72b9b73eaf72f`.

## V4 support rollout (2026-11-10)

The v4 GPU pilot completed on the support split only. Its reported summary is
preserved as emitted by the job; no held-out tasks were accessed.

- Run: `20261110T144247Z`; config: `terminal-slm-tiny-aya-support-rollout-v4`.
- Tasks: the ten preselected support IDs listed above.
- Reported benchmark contract passes: 1/10; automated successes: 1/10.
- Reported failure categories: `agent_loop_guard` 7, `invalid_tool_call` 1,
  `task_contract_failure` 1. The remaining record was the reported pass.
- Training-eligible candidates: 0/10; no training was run.
- Reported average non-null reward: -0.034 over five scored candidates.
- Artifacts: `research/results/tiny-aya-support-rollouts/20261110T144247Z.jsonl`,
  `.summary.json`, `.metadata.json`; cloud log
  `research/results/tiny-aya-support-rollouts/20261110T143917Z.log`.

### Trace audit and contract revision

The job's B006 record reports a pass, but its trajectory shows `read_file` was
called only for line 13, which contains the function signature. It did not
inspect the function body, and the final response describes behavior not
established by that output. The v4 artifact's support-task hash is
`1294a4cd07bb3d27f759f30c572b944993e08cf58e64554929622779732c1035`; the
current local taskset hash differs because B006 was subsequently strengthened
to require its `read_file` output to contain body evidence (`ROUTES` and
`is_positive_id`). Thus 1/10 is the job's original reported result, while the
trace does not satisfy the tightened B006 contract. Do not rewrite the raw
artifact or count B006 as verified factual success under that revised rule.

The repeat-action guard stopped seven loops, but v4 produced no validated code
change. All ten candidates remain ineligible for training. The next useful step
is to inspect the remaining v4 traces and complete run-scoped factual review;
then decide whether a fresh support-only pilot is worthwhile after a concrete
behavior change. Do not start SFT from these candidates.

The v4 review sidecar is
`research/results/tiny-aya-support-rollouts/20261110T144247Z.review.jsonl`.
Rescoring against the current support contracts produced
`20261110T144247Z.reviewed.jsonl` and its summary: 0/10 current contract passes,
0/10 automated successes, and 0 training-eligible candidates. The sidecar has
all ten dispositions marked rejected; the project owner confirmed all ten on
2026-11-12. The reviewed rescore records `human_review_rejected` for every
candidate, with zero training-eligible candidates. The raw run remains
unchanged.

Seven loop-guard records have
`training_target_excluded: true` and
`training_target_exclusion_reason: harness_generated_loop_guard_response` in
the reviewed copy. Their final fallback answer was inserted by the harness,
not generated by Tiny Aya; do not use it as an assistant target. No training
target conversion exists or ran in this phase, and every record remains
`training_target_eligible: false`.

## Exit gate and next experiment

The first run's ten human dispositions are recorded and rejected examples
remain excluded. V4 has now completed, but its sole reported pass fails the
stricter trace audit described above, and no candidate is training-eligible.
Before Phase 5, complete review of v4 and decide whether another support-only
pilot is justified by a concrete behavior change. Do not train from these
candidate sets. Preserve the held-out benchmark and its frozen baseline.

The reviewer identified intentional nonzero test exits being counted as shell
execution failures, schema-invalid calls being double penalized, conflated
change-gate statuses, and missing sidecar ingestion. The scorer now treats a
completed shell process with nonzero exit as an observation, excludes
schema-rejected actions from execution-failure counts, and separates
`rejected_no_file_change` from `rejected_pristine_validator_passes`.
`apply_support_reviews.py` validates and applies the run-scoped sidecar to a
newly rescored copy; raw records remain unchanged and audit mismatches are
explicit. B006 now requires `grep` and `read_file`; B046 requires its own shell
test invocation and output to show `AssertionError: 10 != 25`.

The review file is
`research/results/tiny-aya-support-rollouts/20261109T070739Z.review.jsonl`.
Its recorded scores are applied only if they match the corrected score and the
review confirms agreement. Since the B006/B046 contracts were tightened after
the run, their old results cannot pass those new contracts without the required
evidence. This is a corrected audit of the existing support run, not a rerun.
