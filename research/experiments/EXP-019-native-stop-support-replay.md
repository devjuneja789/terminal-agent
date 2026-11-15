# EXP-019: Versioned live support replay with the native stop union

## Fixed procedure before execution

Run the full terminal-agent loop on B021/B026/B041/B046, with fresh base and
the existing EXP-012 weighted adapter. Apply `[3,6,261001]` to both, preserving
upstream EOS 3, tokenizer turn EOS 6 and chat END_RESPONSE 261001. No new
training, corrective demonstrations, held-out data or source answers enter the
cloud inference bundle.

Derive the replay configuration from EXP-012's identical weighted-adapter replay.
Change only its config ID and opt-in replay fields for stop IDs, stop-policy
version, source preflight lineage and generation telemetry. Preserve task bytes,
prompt/schema hashes, model/adapter revisions, quantization, runtime pins, seed,
generation parameters, agent limits, tools, validators and retry/loop behavior.
Legacy configurations without explicit stop IDs still stop on tokenizer EOS 6.

GPU-only execution on HF T4, capped at 30 minutes. Run eight isolated episodes,
capture complete prompts/model outputs/tool actions/results/final answers and
independent validation, and recover all eight records through bounded chunks.
An opt-in transparent generate wrapper records token IDs, actual configured
stop IDs, closing-token counts and cap hits, forwarding arguments and outputs
unchanged. It captures recovery generations as well as regular model turns.

## Validation and interpretation

Recompute task/validity/step/token/latency/failure aggregates from raw records.
Verify every generation uses the same union and ends at a configured closing
or the unchanged length cap. Inspect file changes, validation output and factual
answers, including intentional nonzero tests. Do not execute new training or
admit replay outputs automatically. Keep all frozen historical settings/results
unchanged and report this as a new runtime version, not a silent result repair.

Compare with archived EXP-012 weighted replay, noting fixed-seed/small-sample
limits. Better stopping is a runtime correction; any efficiency improvement
from it must not be attributed to model training. Even if all tasks fail, the
runtime union can be validated independently of model behavior.

Stop after one full paired replay and audit. No training or held-out evaluation
follows automatically. Launcher: `bash scripts/run_tiny_aya_native_stop_replay.sh`.

## Status

Complete. HF Job [6af8da4a23985fb9cd829cf3](https://huggingface.co/jobs/devjuneja789/6af8da4a23985fb9cd829cf3),
run `20261114T220329Z`, config SHA-256
`bc31e9de5ac85c7867873b75a964693f6f28cb8fc93ba4b983bd72297a4d50d8`.
Job status independently confirmed COMPLETED, 291 running seconds. T4-small
listed $0.40/hour; approximately $0.032 for that running duration, not a billing receipt.

## Configuration and artifacts

Tiny Aya revision `af89d219b53ed9b13b8a4645f9c8028973510324`;
weighted adapter revision `1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0`,
subfolder `20261114T160511Z/adapter`. NF4 double quantization, FP16 compute,
eager attention, greedy generation, seed 42, 256 new-token limit, eight agent
steps and 120-second task limit. Pinned Torch 2.11.0, Transformers 5.22.1,
AgentTune 1.1.3, PEFT 0.21.3, bitsandbytes 0.50.2 and other runtime pins
match EXP-012. Tesla T4, 15,828,320,256 bytes, CUDA 13.0, driver 580.178.04.
Full metadata includes the remaining hardware/package fields and prompt/schema
hashes; the audit checks the old config, quantization, tokenizer, all task
contracts/snapshots/validators and initial prompts for equivalence.

Artifacts under `research/results/tiny-aya-native-stop-replay/20261114T220329Z`:
`.jsonl` (all eight complete trajectories and token IDs), `.metadata.json`,
`.summary.json`, `.md`, `.analysis.json`, `.runtime-gate.json`,
`.job-inspection.txt`. The live stream ended after six task records despite
successful job completion; extraction refused the incomplete set. Fetching
the same job's `.recovered.log` recovered all 60 chunks and summary. No new
GPU execution was needed. Both raw log hashes and artifact/source hashes are
in the runtime gate; raw logs remain gitignored.

## Results

| Arm/runtime | Contract passes | Schema validity | Generated tokens, total | Mean tool steps | Mean agent latency (s) |
|---|---:|---:|---:|---:|---:|
| Base, legacy EXP-012 | 0/4 | 83.33% | 3840 | 2.00 | 60.29 |
| Base, native union | 0/4 | 83.33% | 783 | 2.00 | 14.58 |
| Weighted adapter, legacy EXP-012 | 0/4 | 100% | 965 | 3.75 | 23.50 |
| Weighted adapter, native union | 0/4 | 100% | 946 | 3.75 | 23.29 |

Native-stop validity: **38/38 generations used `[3,6,261001]`; all 38 ended
on 261001, with exactly one recognized closing and zero cap hits.** EOS 3 and
6 remain configured/preserved, but neither was emitted. This validates actual
response-boundary stopping and policy wiring, not separate empirical firing
of every possible EOS ID. Base generated tokens decrease 79.6%; adapter 2.0%.
Mean total prompt+generation tokens/task: base 6806.0, adapter 11893.5.
Tool selection accuracy remains 50.0%/63.64%, invalid-call rate 16.67%/0%.
There are 15 base and 23 adapter model calls including finish recovery.

All aggregate metrics recompute exactly from raw records. Four of eight
selected-action lists are byte-equal to legacy; others have wording/argument
differences. Do not claim all decoded outputs or all actions are identical.
Task outcomes, aggregate validity and action counts are unchanged.

## Factual trace review and failures

| Task | Base | Weighted adapter |
|---|---|---|
| B021 | Invalid action, no edit; original `0 != 1` failure remains. | Replaces module with a self-referencing assignment; NameError, not a repair. |
| B026 | No edit; falsely claims the test passed. | Reads but never writes; falsely claims a change and passing test despite `10 != 25`. |
| B041 | No edit; incorrectly blames the requested test expectation. | Two full-file overwrites delete function definitions; NameError remains. |
| B046 | Runs expected failing test and reads source, but explanation omits ignored-environment cause; also invalid line argument. | Observes expected failure and reads source but does not explain the missing environment read/default fallback. |

The adapter executes three writes across B021/B041, preserving neither module.
Both models change no files on B046; its nonzero test is intentional evidence,
not an executor fault or automatic failure. Its answer still fails the frozen
explanation contract. All model finals are preserved verbatim, including false
success claims. Empty base B021 final has no model answer. Nothing here is
automatically admitted, approved by the owner or added as a training target.

## Conclusion and next experiment

**Runtime stopping gate passes; task success remains 0/4 in each arm.** Native
stopping removes unnecessary generation from the base without establishing
better repair behavior. Historical speed/token advantages of the adapter are
partly confounded by ignored base response closings; do not describe them all
as learning gains. This small support replay is not held-out evaluation.

Save `configs/tiny_aya_native_stopping_v1.json` as a validated opt-in profile;
the EXP-018 candidate remains an immutable pre-validation proposal. Frozen
baseline and historical replay configs/results stay unchanged. Any future
comparison must explicitly version/select the same native profile in both arms.
Next: wire only the 15 admitted corrective continuation targets into one bounded
training experiment, retaining label-mask/weight gates and this stopping profile
for its paired support evaluation. No training or held-out access occurred here.

Local verification: 137 tests run (134 pass, three optional Torch checks skipped),
including three new native-stop tests; eight focused native/paired replay tests
also pass. Independent raw-record/config/token audit passes and shell syntax/
`git diff --check` checks pass. No user action is needed for this completed gate.
