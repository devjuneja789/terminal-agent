# Support reference traces

`gold_support.jsonl` contains six assistant-authored canonical traces built
only from `train-support` tasks B006, B016, B021, B026, B041, and B046. The
builder does not call a model. It executes the declared tools in isolated
copies of the pinned miniapp snapshot and runs each task's validator. The
records include the system/user/assistant/observation message sequence and the
actual tool and validator outputs.

The original file preserves the pre-review **reference candidates**. Each original row has
`human_review_status: pending` and `training_eligible: false`. A project
maintainer must check the task intent, tool arguments, observations, final
answer, and validator outcome before changing either field. Do not train from
raw model rollouts or from harness-generated stop/recovery text.

All sources are in the fixed `train-support` split. No held-out task, trajectory,
answer, or validation result was used to author these examples. Task IDs and
snapshot hashes are retained in every record. Regenerate with:

```sh
python3 -m training.agenttune.build_gold_support --output data/sft/NEW_REFERENCES.jsonl
```

The builder refuses to produce a row if an expected tool fails, a code-change
example does not change a task file, the read-only example changes task files,
or the task's validation exit code is unexpected. The separate
`reference_audit` object records the observed evidence. These traces are not
yet converted to an AgentTune trainer-specific dataset; that conversion should
occur only after maintainer review.

## Owner review recorded on 2026-11-13

The original file is preserved unchanged. `gold_support.review.jsonl` records
the owner's decisions with a SHA-256 of each original record;
`gold_support.reviewed.jsonl` applies those decisions to all six copies.
`gold_support.approved.jsonl` contains only approved G-B021-v1, G-B026-v1,
G-B041-v1, and G-B046-v1. These four are marked training-eligible. B046's
intentional test failure is valid evidence for its explanation task.

Rejected G-B006-v1 and G-B016-v1 remain ineligible. Their corrected replacements
are in `gold_support.regenerated-v2.jsonl`: G-B006-v2 reads lines 11–33;
G-B016-v2 searches the exact setting `REQUEST_TIMEOUT_SECONDS`. Real tools and
validators were rerun for those two records. They await a separate owner review
and are not included in the approved file. Reproduce them at a new output path:

```sh
python3 -m training.agenttune.build_gold_support --task-id B006 --task-id B016 --version 2 --output /tmp/gold_support.regenerated-v2.jsonl
```

The builder refuses to overwrite any existing artifact. Approval applies to
the reviewed record's contents; regenerating a trace does not inherit approval.
The approved JSONL retains provenance and audit fields; it is not yet a dataset
converted to the pinned AgentTune trainer format.
