# EXP-018: Pinned corrective label and stopping preflight

## Fixed protocol before execution

No training, optimizer, backward, tool execution or held-out access. Verify all
15 admitted corrective targets with actual TRL 1.7.1 preprocessing/collation in
the pinned HF T4 runtime (Transformers 5.22.1/Torch 2.11.0). Keep raw source,
owner receipt, applied reviews and admission hashes fixed. Exclude both
audit-only records and all archived assistant prefix turns from supervision.

Require exact generation-prefix/input matching, masked labels on the entire
prompt, one contiguous completion suffix, decoded target equality, no truncation
at 4096 tokens, and supervision of END_RESPONSE (261001) followed by END_OF_TURN
(6). Test the existing 8x/8x/4x weighting annotation, with both closing special
tokens at weight 1, without computing gradients or updating an adapter.

The weighted source helper previously required line_count metadata, which the
runtime strips. It now also accepts a nontruncated read from line 1 returning
fewer lines than requested (EOF), or a read with no end limit. Exact-length
slices without metadata remain unproven and are rejected. Compare all old gold
spans with EXP-012's saved annotations to ensure this compatibility change does
not alter old targets or weights. No reviewed messages are rewritten.

Inspect the official pinned model/generation/tokenizer JSON settings. Compare
two EOS policies on identical saved EXP-014 full-source histories, with fresh
base and existing weighted adapter: legacy `[6]` and diagnostic `[261001, 6]`.
Use seed 42, NF4/FP16/eager, greedy generation, cache enabled, 256-token cap and
no prompt trimming. Three histories x two policies x two models = twelve
generations. Reproduce legacy controls and compare candidate tokens with the
legacy prefix through its first recognized closing token.

Capture all 15 label and 12 generation records, native settings, token IDs,
weight maps, schema validity, cap hits, latency and provenance. This is a
stopping/label diagnostic, not a live agent benchmark. No production/frozen
inference configuration is changed. Stop after one 30-minute-capped HF job and
independent audit; do not begin training automatically.

## Status

Completed one HF T4 preflight and independent artifact audit.
Launcher: `bash scripts/run_tiny_aya_corrective_preflight.sh`.

## Provenance

- HF job `6af8d45b23985fb9cd829585`, run `20261114T213811Z`.
- Config SHA-256 `036b0a86172b84843fe0ccb27f9bedfd8203a771ccde342f81faeabcea8696f4`.
- Admitted row SHA-256 remains
  `03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b`.
- All 27 records recovered: 15 complete label/weight audits and 12 generation
  comparisons. Raw token IDs, prompts, labels, upstream JSON file hashes,
  packages, GPU identity and independent analysis are in
  `research/results/tiny-aya-corrective-preflight/20261114T213811Z.*`.
- No optimizer was created, no gradients accumulated, and no training, tool
  execution or held-out access occurred. Model inference ran only on HF CUDA.

## Label and weighting results

All fifteen rows pass the actual TRL collator audit. Generation prompts match
the masked prefixes exactly; every earlier assistant/user/system token is
masked. Completion labels form one contiguous suffix, reproduce exactly the
reviewed JSON action, and include END_RESPONSE (261001) then END_OF_TURN (6).
The two native closing special tokens retain weight 1. Critical tool-name,
changed-code and JSON-closure spans map to supervised tokens for all rows.
There are three complete-file edit targets; no audit-only record is present.

Maximum sequence length is **2490**, below 4096. All full-template IDs match
trainer IDs without truncation. The metadata-free complete-read compatibility
change passes tests for conservative EOF detection, partial/truncated rejection,
and unchanged old gold weighting spans. No source text or target was rewritten.

The separate data gate is `data/sft/tiny_aya_corrective_v1.preflight.json`.
Creation-time admission manifests remain historical; this gate records the
completed pinned audit against the exact admitted row hash.

## Native setting verification

The pinned upstream model and generation JSON both specify EOS **3**, BOS 2,
PAD 0. The tokenizer instead names END_OF_TURN (**6**) as its EOS, and the chat
template appends END_RESPONSE (**261001**) before END_OF_TURN. These are three
distinct IDs; upstream generation JSON does not itself specify 261001.

The legacy runner overwrites generation EOS with tokenizer EOS 6 alone.
Comparing that legacy policy against response-or-turn stopping confirms that
ignoring END_RESPONSE causes substantial unnecessary generation in these base
cases. It does not show that every model failure is a stopping failure.

## Matched stopping results

All six legacy controls reproduce the saved decoded outputs. For every model/
case pair, the candidate's token sequence is exactly the legacy sequence through
the first response-or-turn closing token. All six candidate generations stop
at 261001, and every decoded action is unchanged.

| Metric, three fixed prompts per column | Base `[6]` | Base `[261001,6]` | Adapter `[6]` | Adapter `[261001,6]` |
|---|---:|---:|---:|---:|
| Generated tokens | 768 | 121 | 245 | 242 |
| Cap hits | 3 | 0 | 0 | 0 |
| Schema-valid actions | 3/3 | 3/3 | 2/3 | 2/3 |
| END_RESPONSE occurrences | 643 | 3 | 3 | 3 |
| Mean generation latency, seconds | 15.48 | 3.10 | 6.70 | 6.60 |

The base token reduction is 84.2% on these three saved prompts, eliminating
all three cap hits. Fixed arm/policy ordering, warmup and output-length changes
limit latency comparisons. This is not an end-to-end speed claim. The unchanged
actions still include the adapter's invalid B021 JSON and repeated B026 read;
there is no new code-correctness or task-success improvement.

Historical base/adapter efficiency comparisons mix learned stopping behavior
with a runner that ignores the native response boundary. Do not present all
previous token/latency reductions as better planning from SFT. Preserve their
raw values, but rerun both models with a consistent corrected runtime before
making new hackathon efficiency claims.

## Candidate and next gate

Do not drop upstream EOS 3 when correcting chat stopping. The inactive proposal
`configs/tiny_aya_stopping_candidate.json` preserves upstream 3 plus turn 6 and
response 261001. Its `[3,6,261001]` union is not directly GPU-tested here; the
directly tested policy is `[261001,6]`. None of the six legacy generations
contains token 3 before the first chat closing, so adding 3 would recognize the
same first boundary on the recorded prefixes. That is an offline observation,
not a substitute for versioned live support validation.

Next run a separately versioned paired support replay using the full native
stop union on both base and adapter, then wire the audited corrective export
into a bounded training profile. Keep benchmark tasks and all frozen historical
settings/results untouched. No new owner review is needed for unchanged data.
No training or follow-up GPU experiment was launched in this step.

Verification: 134 local tests pass with three optional Torch skips. Five new
tests cover label boundaries, both closings, source-completeness evidence,
old-span parity, bundle isolation and full chunk recovery. Shell syntax and
Git whitespace checks pass.
