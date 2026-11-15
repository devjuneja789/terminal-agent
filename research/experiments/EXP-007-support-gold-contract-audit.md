# EXP-007 — Support contract audit and verified reference traces

**Status:** Four v1 traces approved; two corrected v2 traces pending review. No model
inference or training was run.
**Split:** `train-support` only.

## Motivation

The V5 rollout exposed weak keyword-only contracts. B016 passed without
answering whether the function reads the environment variable. B046's
automatic checks did not require inspecting the implementation. B041 produced
no `write_file` call and no source change. Therefore the rollout's apparent
passes were not reliable evidence of task completion.

## Changes

- Strengthened B016 to require an answer that `request_timeout()` does not read
  `REQUEST_TIMEOUT_SECONDS`, plus a `read_file` observation from `shop/config.py`
  containing the function and its return statement.
- Strengthened B046 to require the agent's own failing test output, a read of
  the relevant implementation, and an explanation tying `10 != 25` to the
  function returning `DEFAULT_TIMEOUT_SECONDS` without reading the variable.
  The evaluator also enforces that this read-only task changes no task files.
- Strengthened B021, B026, and B041 to require both `write_file` and the agent's
  successful test observation. B026 separately requires the initial expected
  failure and the successful rerun.
- Extended task evidence matching with exact `arguments_match` checks. This
  prevents reading an unrelated file from satisfying source-evidence criteria.
- Added six canonical reference traces for B006, B016, B021, B026, B041, and
  B046. The builder executes the actual workspace tools and validators against
  disposable copies; it makes no model calls.

## Verification

The reference builder recorded these outcomes:

| Task | Tools exercised | Validation exit | Task files changed |
|---|---|---:|---:|
| B006 | grep, read_file | 0 | No |
| B016 | grep, read_file | 0 | No |
| B021 | read_file, write_file, shell | 0 | Yes |
| B026 | shell, read_file, write_file, shell | 0 | Yes |
| B041 | read_file, write_file, shell | 0 | Yes |
| B046 | shell, read_file | 1 (expected) | No |

The B046 failure is a task observation, not a failed tool launch: the process
ran and reported the expected assertion. All six records remain
`training_eligible: false` pending maintainer review. No new rollout or
training job was launched.

## Decision

### Owner review (2026-11-13)

The owner approved G-B021-v1, G-B026-v1, G-B041-v1, and G-B046-v1 after
reviewing the supplied traces and audits (without independently running tests).
G-B006-v1 was rejected for reading 11–21 instead of the prompted 11–33 range;
G-B016-v1 was rejected for searching an alias instead of the exact setting.
Both answers were factually correct; the rejections concern instruction compliance.
The decisions and hashes are stored in `data/sft/gold_support.review.jsonl`.
The approved-only file contains four eligible references. The original file
is unchanged. Both rejected examples were corrected and executed as v2 records
in `data/sft/gold_support.regenerated-v2.jsonl`, with local validation exit 0.
Their new contents still require review; approval was not transferred from v1.

Do not resume rollout generation yet. Convert only the four approved references
from `data/sft/gold_support.approved.jsonl` to the pinned AgentTune SFT format,
then conduct a small support-only training smoke before any held-out evaluation.
