# EXP-017: Apply owner review and separately admit corrective data

## Authorization and review provenance

The owner approved all six EXP-016 records, explicitly separating review from
admission and keeping the last two permanently audit-only. The original receipt
recorded those decisions without applying them. The subsequent user instruction
authorized proceeding to the next step and applying review changes as needed.

This step applies the six decisions to the sidecar and separately admits only
the four task-completing records. It does not change any decision or reason.
Original candidates, their creation manifest, readable packet and original
owner receipt remain immutable. Preserve the old pending sidecar byte-for-byte
as `support_corrective.v1.review.pending.jsonl`.

## Admission checks

- Verify all six candidate hashes, unique IDs, owner receipt/batch hash and the
  exact pre-application sidecar hash. Reject missing approvals or scope changes.
- Enforce permanent exclusion of both audit-only records, even though their
  factual accuracy was approved. Only B021/B026/B041/B046 support tasks qualify.
- Verify current task contract, snapshot, source replay hash and exact archived
  prefix. Re-execute all four admitted transcripts in fresh disposable snapshots.
  Every action has valid native schema; only expected file operations and the
  recorded named/upper-bound tests may run. Compare exit outcomes and file output
  against reviewed evidence; preserve fresh results separately.
- All three repairs pass their named validator and existing upper-bound check.
  B046 retains the intentional failing test, explains its real cause, and changes
  no task file. Expected nonzero test exits are not executor failures.
- Export only proposed continuation indices after the archived prefix. Create
  one corrected completion per row and mark the whole prompt as context to mask
  during completion-only training. Never use generic all-assistant extraction.
- Deduplicate exact prompt/completion pairs and reject conflicting targets for
  identical prompts. Do not merge the old gold dataset automatically.

## Results

Four admitted demonstrations, 15 unique action targets, zero duplicates and
zero exact overlaps with the old gold action rows. Both audit-only reports are
excluded. All four transcripts revalidated and all 15 structural target-boundary
checks pass. No held-out records, inference, training or benchmark edits occur.

Corrective row SHA-256:
`03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b`.

Artifacts:

- `data/corrective/support_corrective.v1.review.jsonl`: applied six decisions.
- `data/corrective/support_corrective.v1.admitted.jsonl`: four derived records.
- `data/corrective/support_corrective.v1.admission.json`: admission/provenance.
- `data/corrective/support_corrective.v1.revalidation.jsonl`: real execution evidence.
- `data/sft/tiny_aya_corrective_v1.jsonl`: dedicated continuation-only export.
- Adjacent `.manifest.json` and `.boundaries.jsonl`: hashes and structural audit.

These are verified assistant-authored demonstrations, not new model successes.
The original source status remains creation-time pending in immutable candidates;
the applied sidecar and admission record describe the new state.

## Tokenizer/stopping preflight finding

Local packages are Transformers 5.19.0 and Torch 2.14.0+cu126, while the frozen
cloud runtime uses Transformers 5.22.1 and Torch 2.11.0. Do not represent a local
tokenization check as the pinned TRL label audit. No package was installed and
no weights were loaded locally.

A read-only inspection of the cached tokenizer configuration at the pinned model
revision identifies token 6 as `<|END_OF_TURN_TOKEN|>` and token 261001 as
`<|END_RESPONSE|>`. The assistant chat template appends END_RESPONSE followed by
END_OF_TURN. Existing inference code overrides generation EOS with only the
tokenizer EOS (6), while saved base generations repeatedly emit 261001 and hit
the 256-token cap. Preserve the static tokenizer/template and prediction evidence
in `research/results/tiny-aya-corrective-admission/stopping-preflight.json`.

This identifies a stopping configuration issue to verify; it does not prove a
task-success gain, justify overwriting old results, or invalidate the admitted
JSON action text. No stop setting or training label was changed in this step.

## Next gate

Run a pinned cloud preflight before any more training: verify official generation
settings, both assistant closing tokens, exact prompt/completion boundaries,
context masking, sequence lengths and the intended EOS labels on all 15 rows.
Any inference correction requires a separately versioned support comparison on
both base and adapter, preserving frozen historical runs. Existing training
launchers still use old gold data; do not run them expecting this corrective set.

No further owner review is needed for these same four admitted demonstrations.
No new training experiment is launched here. Verification: 129 local tests,
three optional Torch skips, and four focused admission tests pass. The focused
tests also pass after applying the real sidecar, using the preserved pending
backup as their fixture. Git whitespace checks pass.
