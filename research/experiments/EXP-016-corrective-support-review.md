# EXP-016: Corrective support demonstrations for owner review

## Purpose and boundaries

Prepare assistant-authored corrections inside actual failing conversations,
before any more training. These are supervised demonstration candidates, not
model successes, independent benchmark results, or owner-approved examples.
No model inference or training runs in this preparation.

Source: weighted support replay `20261114T161125Z`. The six records cover only
the previously approved support tasks B021/B026/B041/B046. Each record binds
the original replay record, snapshot, task contract and exact context prefix
with hashes. Re-execute prefix tools in a disposable snapshot to reproduce its
file state and exit outcomes, while preserving the actual archived messages.
Fresh prefix-replay outputs are audit evidence, not replacements for that history.

## Review batch

| ID | Correction | Status |
|---|---|---|
| C-B021-read-complete-v1 | After partial read, read complete file, preserve upper cap while fixing lower bound, run both named and upper-bound tests | Task-completing candidate |
| C-B026-choose-write-v1 | After failing test and full read, choose complete-file write rather than repeated read, then verify | Task-completing candidate |
| C-B041-preserve-file-v1 | After partial read, inspect complete module and change only default constant, verify both checks | Task-completing candidate |
| C-B046-explain-evidence-v1 | Explain intentional 10 != 25 failure using existing test/source observations, with no edits | Task-completing candidate |
| C-B021-truthful-failure-v1 | After actual destructive write/test failure, admit task is not fixed and report the NameError | Audit-only; excluded from training |
| C-B041-truthful-failure-v1 | After actual incomplete module write/test failure, report missing function and no passing verification | Audit-only; excluded from training |

All six require owner review. The first four propose 15 action targets;
the last two propose zero targets and are never counted as successful repairs.
They show truthful alternatives to false completion claims without teaching
premature abandonment as successful task behavior. There are four independent
tasks, not six new tasks; overlapping continuations are not dataset diversity.

## Execution and target safety

- Read complete original source before generating any corrective full-file write.
  New write contents derive from that visible read, not hidden snapshot knowledge.
- Validate each native AgentTune action and execute tools locally in temporary
  workspace copies. Use the actual runtime bounded-result normalizer and JSON
  serialization, including its empty metadata. Preserve complete real results.
- All three repair continuations pass their named validator and the existing
  page-size upper-bound check. The explanation task intentionally retains a
  nonzero test exit and satisfies the explanation contract without changes.
- Both failure-report records retain a failed task, perform no corrective file
  change, and remain permanently audit-only under this batch's admission policy.
- Archived assistant turns are context only, including incorrect actions.
  Explicit target indices begin after `context_message_count`; inspection export
  creates one completion per corrected action with the entire prefix masked in
  a future completion-only trainer. No prefix turn becomes a training completion.
- Review candidates and proposed rows are training-ineligible. Existing gold
  files, reviews, benchmark contracts, split and training scripts remain unchanged.
  No approved dataset or cloud training bundle is generated here.
- Builder refuses to overwrite the candidate, sidecar, manifest or review packet.
  Hash-bound reviews must refer to this exact generated batch. Do not rebuild it
  after review and silently keep old approvals.

## Artifacts and owner action

- `data/corrective/support_corrective.v1.jsonl`: complete candidates and audits.
- `data/corrective/support_corrective.v1.review.md`: readable review packet.
- `data/corrective/support_corrective.v1.review.jsonl`: six pending review decisions
  bound to source-record hashes. Set each decision to approved/rejected and add
  reviewer/reason, or provide those six decisions in conversation.
- `data/corrective/support_corrective.v1.manifest.json`: counts and provenance.

Review means correctness, instruction compliance, complete-file preservation,
truthful conclusions and evidence. Approving an audit-only report's accuracy
does not admit it to training. Training admission, duplicate handling and any
new bounded experiment remain separate work after this review.

Reproduction (use a new destination; original batch must stay immutable):

```sh
python3 -m training.agenttune.build_corrective_support --output /tmp/corrective-recheck.jsonl
```

## Verification

125 local tests pass with three optional Torch skips. Four focused tests execute
and check the full batch: exact archived-prefix
preservation, continuation-only targets, runtime-compatible observations,
named/regression checks, honest expected failure, review hashes and overwrite
protection. All three repairs pass named and upper-bound checks; the explanation
passes its contract with an intentional nonzero test exit. The batch manifest
reports zero training-eligible records. Readable packet rendering is also checked.
