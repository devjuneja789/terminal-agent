# EXP-026: Joint-300 from actual support task starts

The user explicitly requested another bounded attempt after the completed
submission candidate and evaluation. Preserve EXP-024/025, their candidate
freeze, benchmark and results. This is additional support-only development;
the earlier claim that no work follows final evaluation is historical as of
this new request. No held-out task or failure supplies a new target, prompt
or reward. Further experiments cannot make the already observed held-out set
newly unseen.

## Hypothesis and scope

Joint-300 learned the 15 reviewed intermediate action choices, but its full
agent performance on the four approved support tasks has not been measured.
Run B021, B026, B041 and B046 from their original task starts, without supplied
demonstration histories, on both untouched base and joint-300. Use exactly
the existing native-stop support runtime: NF4/FP16, fixed prompts/schemas,
seed 42, greedy 256-token generations, eight steps and 120 seconds per task.
No inference change, training, new examples or automatic admission. One T4
job with the existing 30-minute launcher cap; no repeated seed search.

## Decision rule declared before execution

Audit all eight trajectories and actual edits. Require correct complete-file
changes, task-validator passes and relevant existing regression checks before
describing a repair as successful. Expected nonzero test outcomes on B046
are evidence, not execution faults. Inspect factual claims against observations.
Report these as known training-support tasks, never held-out generalization.

If at least two of the three edit tasks are verified successful, use this as
a narrowly scoped support demonstration and stop new optimization. Otherwise,
the next bounded intervention should teach complete trajectories from fresh
task starts, using the four previously approved gold references together with
the admitted corrections. First validate their provenance and initial-state
coverage. Do not simply extend the same continuation-only training run.

The pre-existing submission package remains available regardless of outcome.

## Execution and coverage audit

HF job `6af9982223985fb9cd8340c5` was submitted under the user's standing GPU
authorization. It spent 120 seconds scheduling before starting. Eighteen
focused local tests pass (one optional AuditKIT test skipped in the system
Python); no inference runs locally.

The independently verified approved gold export contains 16 action targets,
including four first actions on fresh user tasks. The admitted corrective
export contains 15 actions and **zero** fresh-task first actions. Their union
contains 31 distinct prompt/completion pairs, all from the same four approved
support tasks. This confirms missing initial-state supervision in joint-300;
it does not establish that mixing these datasets improves outcomes. No new
training admission or training has occurred. See
`research/results/task-start-coverage.json` for exact source hashes.

Completed as run `20261115T113513Z`: **0/4 base, 0/4 adapter**. All eight
records were recovered and runtime, split, contract and aggregate audits pass.
Adapter B021 emits malformed/multiple actions. B026 issues unsuitable shell
commands, creates `shop/environment.py` and falsely claims the repair passed;
the requested configuration implementation remains unfixed. B041 reads an
insufficient range and falsely claims a default change. B046 observes evidence
but incorrectly blames an unset environment variable instead of the function
ignoring it. Zero successful write_file calls. No verified repair demo exists.

The declared success gate is not met. EXP-027 proposes the complete-trajectory
curriculum; do not spend another identical continuation-only training budget.
