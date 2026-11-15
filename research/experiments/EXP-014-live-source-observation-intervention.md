# EXP-014: Complete source inside the actual failing conversation

## Question and fixed protocol

EXP-013 showed one exact complete-file edit under an approved history. Test
whether complete source alone changes the next action within the actual
failing weighted conversations, without replacing earlier actions or adding
instructions. No training, new demonstrations or held-out access is allowed.

Use EXP-012's weighted adapter, commit
`1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0`, and untouched pinned Tiny Aya base.
Both models receive the same three archived adapter histories from
`20261114T161125Z`, immediately after their first config.py read. This is a
conditional cross-model diagnostic, not each model's own end-to-end rollout.

For each history, run two greedy next-action predictions:

- **original:** exact archived messages, byte-for-byte.
- **full_source:** same messages, changing only the last JSON observation's
  stdout to the complete original 18-line file. All roles, instructions, earlier
  turns, read arguments and other observation fields remain unchanged.

B021/B041 originally read 4/2 lines. B026 already read all 18 lines: its two
conditions must be byte-identical and serve as an identity control. The full
observations are explicit counterfactuals; the original read requests remain
unchanged. Do not claim a tool actually returned these augmented observations.

Freeze seed 42, pinned EXP-012 package versions, NF4/FP16/eager inference and
the live generation settings (greedy, cache enabled, 256 new tokens, no prompt
trimming). Score all twelve predictions: 2 models x 3 histories x 2 conditions.
Compare weighted original outputs against the archived text to check faithful
reproduction. Require identical B026 inputs and report output equality.

## Checks and interpretation

Reuse EXP-013's strict action parsing, whole-module preservation and exact-gold
comparison. Apply only valid writes to the expected config.py path in disposable
snapshots; run the unchanged named validators. Also run the previously identified
B021 upper-bound test separately in both conditions; it does not change the
frozen task contract. Do not execute generated shell commands.

Report original/full comparisons by task/model, control reproduction, preservation,
named tests, upper-bound regression, latency, tokens and cap hits. Preserve
all prompts, input IDs, outputs and provenance. No diagnostic output is admitted
as training data. A next-action change is not a completed agent task or held-out
gain; no automatic continuation, training extension or prompt tuning follows.

## Status

Completed one HF T4 inference-only job and isolated local validation.
Launcher: `bash scripts/run_tiny_aya_observation_intervention.sh`.

## Provenance and control checks

- HF job `6af8a66223985fb9cd8263c0`, run `20261114T182159Z`, Tesla T4.
- Config SHA-256 `7acac0e7b72ee79c3df32f2e6b772bcf3a20c23c67cb14455210d5254f1f3315`.
- Same pinned model, adapter, packages and quantization as EXP-012. Generation
  settings exactly match that run. No model inference ran locally.
- All twelve raw predictions, complete messages, input IDs and generated IDs
  were recovered; local audit independently reconstructs the matched inputs.
- Both models have identical input token IDs for each case. Only the final
  observation stdout changes in the two treated histories. Every earlier
  message and observation field is preserved.
- The adapter reproduces **all three archived original outputs exactly**.
  B026's two inputs/token sequences and outputs are identical in both models.
  This supports interpreting within-history differences as observation effects
  under this fixed model and decoding configuration.
- All three pristine task validators fail as expected, with no file changes.
  Only three valid predicted writes are applied to disposable snapshots: two
  original adapter writes and the full-source B041 write. No generated shell
  commands or invalid actions execute. Named validators remain unchanged.

## Results

### Adapter next actions

| Task | Original observation | Complete-source observation | Conditional validation finding |
|---|---|---|---|
| B021 | Writes a self-referencing one-line assignment; NameError | Proposes a truncated function snippet inside malformed JSON | Rejected without execution; no valid repair |
| B026 | Reads the same file again | Identical read, as expected for the identity control | No write to validate |
| B041 | Replaces module with constant plus undefined function call; NameError | Writes constants and both complete original functions, with default set to 15 | Named timeout test now passes; unused import and module docstring are removed |

B041 is a real conditional functional improvement: the requested default is
updated, both functions' ASTs are preserved, and the unchanged named validator
passes after previously failing. It is **not** exact minimal file preservation:
the unrelated `import os` and module docstring are omitted. Their removal is an
edit-scope defect; the recorded test does not show a functional regression from
those omissions. Do not turn a strict preservation failure into a claim that
the named repair itself failed.

B021's full-source output closes its structures incorrectly. The proposed
content also lacks `def` and uses a lower bound of zero, but it is not applied
or repaired by the harness. The original write's named and upper-bound tests
both fail on its import-time NameError. No JSON-normalization workaround or
replacement success result is introduced.

### Aggregate diagnostic counts

All columns below have three next-action decisions, including B026's unchanged
control. Only B021/B041 actually receive different source content.

| Metric | Base original | Base full source | Adapter original | Adapter full source |
|---|---:|---:|---:|---:|
| Schema-valid actions | 2/3 | 3/3 | 3/3 | 2/3 |
| Schema-valid writes to expected path | 0 | 0 | 2 | 1 |
| Strict surrounding-module preservation | 0 | 0 | 0 | 0 |
| Exact approved file content | 0 | 0 | 0 | 0 |
| Named validators run | 0 | 0 | 2 | 1 |
| Named validators passing | 0 | 0 | 0 | 1 |
| Generated tokens | 768 | 768 | 123 | 245 |
| Token-cap hits | 3 | 3 | 0 | 0 |
| Mean generation latency, seconds | 15.62 | 15.73 | 3.85 | 7.00 |

Base proposes shell commands, not writes, in both conditions. Its six outputs
all hit the frozen 256-token cap; five end in repeated token ID 261001 and
decode/strip to a valid single action. The token's identity was not inspected.
Preserve raw generated IDs and cap flags.
No claim is made about what a longer base generation would do. The adapter's
failures are not caused by the token cap in these cases. Fixed ordering and
different generation lengths limit latency comparisons.

## Conclusion and next decision

Complete source alone changes the actual failing adapter histories and fixes
the named default-timeout behavior in B041. It does not make writing reliable:
B021 regresses to invalid JSON, B026 still chooses another read, and B041 still
removes unrelated file contents. This is a narrower and more defensible result
than either "source does not help" or "full source fixes the agent."

EXP-013's exact B041 gold-history result does not transfer unchanged to this
live-history intervention. Differences beyond source availability still matter.
The remaining candidates include conversation/observation serialization,
earlier action arguments and insufficiently learned edit/stopping behavior;
this experiment does not isolate those causes.

Do not start longer training or present these conditional checks as a new
benchmark success rate. The last full agent replay remains EXP-012's 0/4.
Before changing the objective again, compare runtime observation serialization
with the reviewed demonstration serialization under matched live histories.
Any resulting data or harness change needs its own version and preserved
before/after comparison. No follow-up experiment or training was launched here.

Artifacts: `research/results/tiny-aya-observation-intervention/20261114T182159Z.*`,
including raw JSONL, configuration/runtime metadata, twelve preservation audits,
validator output, analysis, source hashes and raw-log provenance. No output is
an approved training target. No held-out records or benchmark files changed.

Local verification: 118 tests with three optional Torch skips; four focused
tests cover exact input preservation, identity control, bundle isolation and
complete chunk recovery. Shell syntax and Git whitespace checks pass.
