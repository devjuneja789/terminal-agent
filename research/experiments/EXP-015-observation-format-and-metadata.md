# EXP-015: Runtime versus reviewed observation format and metadata

## Question and static finding

Does an observation-format mismatch explain the adapter's dependence on approved
histories? Compare actual reviewed messages with the runtime serializer before
assuming a mismatch or spending more on training.

The finding is **payload metadata, not JSON layout**. All twelve reviewed
observations already use `Tool result (JSON):` and canonical two-space-indented
JSON with the runtime's escaping. Given the same payload, TinyAyaPolicy's
runtime serializer reproduces all twelve byte-for-byte. The earlier proposed
serialization hypothesis was unverified; do not report a discovered layout bug.

`evaluate_heldout._bounded_result` unconditionally assigns `metadata = {}`.
The reviewed references retain metadata on four read_file results and three
write_file results; five shell observations already have empty metadata. With
the large nontruncating audit limit, metadata is the only payload difference.

## Fixed metadata-only comparison

Start from EXP-014's three full-source adapter histories, not the gold histories.
Both base and the pinned weighted adapter receive the same paired inputs:

- **runtime:** exact EXP-014 full-source messages, metadata still `{}`.
- **reviewed_metadata:** change only the last observation's metadata to the
  reviewed read result's `{"path":"shop/config.py","line_count":18}`.

The source text is complete and identical in both conditions. Instructions,
roles, prior actions, read arguments, other observation fields, serializers,
model revisions, NF4/FP16 inference, seed 42 and live generation settings stay
fixed. B026 now receives a genuine metadata intervention; it is no longer an
identity control. Preserve and independently validate all six inputs/twelve
predictions and compare the three adapter runtime controls with EXP-014 outputs.

Restore truthful file metadata, not expected code or gold answers. GPU bundles
contain only inputs, configuration and inference code. No training, backward,
held-out access, benchmark or production-harness edits are permitted.

## Validation and stopping rule

Reuse strict action parsing, complete-module preservation, exact-gold checks
and isolated application of eligible writes only. Run unchanged named validators
and the already documented B021 upper-bound test separately. Generated shell
commands and invalid actions must not execute. Report conditional edit checks,
not a complete agent success score. Stop after one HF T4 inference job (30-minute
cap) and factual audit, whether metadata helps or not. Do not sweep prompts,
extend training or silently change the runtime normalizer after seeing results.

## Status

Static audit, inference comparison and isolated validation complete.
Launcher: `bash scripts/run_tiny_aya_observation_metadata.sh`.

## Configuration and provenance

- HF Tesla T4 job `6af8c2b629dd05ff0367c1e5`, run `20261114T202246Z`.
- Config SHA-256: `2783903d87882b196f1e821a194fd95ee466f3f4d27d5c0c56a084f090cbd652`.
- Model, adapter, package pins, NF4 quantization and generation settings match
  EXP-014. The weighted artifact remains commit
  `1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0`; no weights are updated.
- All twelve predictions recovered with full messages and token IDs. Both models
  receive identical tokens for each case; independent audit confirms only the
  final metadata field differs within pairs. Source text is unchanged.
- The three adapter runtime controls exactly reproduce EXP-014. All pristine
  change-task validators fail as expected with no changes. No held-out records,
  new training targets or gold answers are included in the GPU bundle.
- Static observations and source hashes are saved in
  `research/results/tiny-aya-observation-metadata/static-format-audit.json`.
  Raw predictions, configuration/runtime metadata, preservation audits, validator
  output and provenance are in `20261114T202246Z.*` in the same directory.

## Results

| Adapter case | Runtime metadata `{}` | Reviewed metadata restored |
|---|---|---|
| B021 | Malformed JSON containing an incomplete function snippet | Exactly the same malformed output; rejected without execution |
| B026 | Reads config.py lines 1–20 again | Exactly the same read despite line_count=18 |
| B041 | Writes default=15 with both functions but drops import/docstring; named test PASS | Writes default=15 followed by default=10; named test FAIL |

For B041, the second assignment wins and resets the timeout to 10. The unchanged
named test confirms `10 != 15`. It still omits the import and module docstring,
so neither variant meets strict complete-file preservation. This is a genuine
regression from EXP-014's conditional functional gain, not an evaluation update.

| Metric, three decisions per column | Base runtime | Base restored metadata | Adapter runtime | Adapter restored metadata |
|---|---:|---:|---:|---:|
| Schema-valid actions | 3/3 | 3/3 | 2/3 | 2/3 |
| Valid writes to expected path | 0 | 0 | 1 | 1 |
| Strict complete-module preservation | 0 | 0 | 0 | 0 |
| Exact approved file content | 0 | 0 | 0 | 0 |
| Named validators run | 0 | 0 | 1 | 1 |
| Named validators passing | 0 | 0 | 1 | 0 |
| Generated tokens | 768 | 768 | 245 | 251 |
| Token-cap hits | 3 | 3 | 0 | 0 |
| Mean generation latency, seconds | 15.81 | 15.58 | 6.86 | 7.06 |

Base decoded outputs are unchanged for all three pairs and propose shell
commands rather than writes. All six base generations hit the unchanged
256-token cap; this limits inference about base behavior beyond the cap.
Adapter outputs change only on B041, and none hits the token cap. Latencies
are descriptive and do not establish a speed comparison.

Only the two eligible B041 predicted writes are applied to disposable snapshots;
the other actions do not supply valid file contents. Invalid JSON and generated
shell commands are never executed. No B021 predicted write is eligible for its
separately specified upper-bound check. Raw failures are preserved.

## Conclusion and next step

The suspected JSON-format mismatch is ruled out for the twelve reviewed
observations: the runtime already uses their serializer. Metadata omission is
real, but restoring it alone does not repair the tested behavior and regresses
one conditional fix. Do not claim it explains the agent's failures or change
the frozen normalizer on the strength of this test.

Stop formatting-only experiments here. Remaining observed errors are malformed
actions, repeated reads, lost module contents, wrong code changes and unsupported
completion claims. A useful next step is a reviewed support-data revision that
demonstrates correct decisions and recovery inside actual runtime histories,
with regression checks and complete-file preservation. That work must use only
support tasks, retain owner review and source hashes, and precede any new training
budget. It was not started in this experiment.

These are single next-action comparisons, not complete task runs. The last
end-to-end support replay remains EXP-012's 0/4. No new benchmark success rate,
generalization gain or training-eligible trajectory is claimed. Benchmark tasks,
approved references and production observation formatting remain unchanged.

Verification: 121 local tests, three optional Torch skips. The three new tests
cover byte-level serializer parity, metadata-only edits and input-bundle isolation;
existing extraction/control tests still pass. Shell syntax and Git whitespace
checks pass. No local model inference ran.
