# EXP-025: Frozen-candidate matched held-out comparison

## Hypothesis

The support-selected joint adapter may improve executable terminal-task
completion relative to untouched Tiny Aya under matched inference. The null
result remains possible; exact-context fitting is not end-to-end success.

## Protocol declared before the final run

Candidate selection follows EXP-024 on the same 15 reviewed support contexts.
The immutable freeze records the model, adapter commit/subfolder, training
result hash and support scores. No selection or training follows held-out
inspection. Evaluate all ten permanently held-out IDs B005, B010, B015, B020,
B025, B030, B035, B040, B045 and B050. Compare task file bytes against the
committed benchmark before staging. No task, validator or reference outcome
changes. Earlier held-out experiments remain historical rather than unseen.

Both arms use pinned Tiny Aya Global, NF4/double quantization with FP16
compute, eager attention, seed 42 per task, greedy decoding, max 256 new tokens,
7000 prompt/8192 total context limits, stop IDs `[3,6,261001]`, the same v5
prompt, tool schemas and observation/loop policy. Eight action steps, 120
seconds cooperative agent time and 120 seconds validation time per task.
Use T4-small for both arms in one job with a 50-minute cap. The pre-existing
native-stop runtime is reused, not tuned for a held-out task. The old FP16
baseline is not this new runtime's control.

The final staging bundle contains only held-out tasks and snapshot/code;
no SFT data or support task file. Shared replay helpers are imported for
tokenizer equivalence, stop validation and telemetry, but no reward scoring
function is invoked. All twenty raw episodes and every returned generation
are captured, including decoded output from a call that exceeds a cooperative
deadline. No missing record is silently dropped.

## Metrics and independent checks

Report deterministic task-contract success, tool selection, argument/schema
validity, invalid-call rate, steps, model turns, tokens, latency and failure
categories. Tool selection follows the existing task-required-tool proxy;
it is not an oracle for the optimal tool on every individual action. Tool
arguments/schema validity is distinct from semantic command correctness.

Recompute contract outcomes and aggregate metrics from the saved raw records.
Run the real pinned AuditKIT AgentEvalRunner over imported AgentTune events.
Its independent recorded-evidence oracle recomputes frozen contracts, including
expected nonzero tests and no-change conditions. Its structural scorers cover
emitted events; they do not replace rejected-output attempt metrics. Preserve
harness-vs-model final-response attribution. Produce a standalone recorded
walkthrough and machine-readable evidence index.

The output-only addition `task_file_changes` exposes already computed evidence
for the independent no-change check. It changes no tool, prompt, validation,
policy or task condition. File diffs exclude ordinary Git/internal log metadata
as defined by the existing runner.

## Status

Completed as HF Job `6af95c1129dd05ff0368583f`, run `20261115T071710Z`.
Joint-300 was frozen at 07:14:40 UTC before staging. All twenty complete
records, identical-config/freeze checks, independent contract recomputations
and aggregate checks pass. Real AuditKIT reports cover ten episodes per arm,
with zero method errors or missing outcomes.

| Metric | Untouched base | Joint-300 |
|---|---:|---:|
| Frozen contract passes | 1/10 | 2/10 |
| Tool selection proxy | 31.25% | 35.71% |
| Argument/schema validity | 94.12% | 94.74% |
| Invalid call rate | 5.88% | 5.26% |
| Average tool steps | 1.30 | 1.10 |
| Average model turns | 2.60 | 2.80 |
| Average input + output tokens | 4112.8 | 4471.8 |
| Average agent latency | 6.926 s | 10.641 s |
| Code-repair/update passes | 0/5 | 0/5 |

Both pass B010's location/signature rubric. Adapter additionally passes B050's
weak test-failure rubric, but its explanation contains incorrect and unrelated
claims. The nominal +10 percentage points does **not** establish an additional
factually reliable completion. No task file changes occur in any episode.
See [the detailed trace review](../../submission/TRACE-REVIEW.md). AuditKIT
reproduces contracts, not a stronger independent semantic judgment.

Base failures: one invalid call, three task-contract failures and five
validation failures. Adapter: one invalid call, three contract failures and
four validation failures. B040's attempted writes target a directory and are
rejected; B045 emits multiple actions and is rejected before execution. Initial
searches often find no evidence, followed by an unhelpful finish. Longer
training fixed conditional choice learning but not reliable runtime transfer.

## Conclusion and next experiment

The pipeline is executable and auditable, and the limited reviewed contexts
can be learned substantially better with the fixed longer budget. Reliable
end-to-end repair is not demonstrated. Small token/step/validity changes do
not establish an efficiency gain; latency and token usage increase.

Stop research for this submission. Preserve v1 tasks and both real outputs;
no held-out-derived targets, rewards or prompt edits. Future work should collect
more diverse reviewed support initial states/action arguments and evaluate on
a separately versioned stronger factual rubric with a fresh matched baseline.
Do not reinterpret this ten-task result as broad model superiority or a
capacity limit. Full machine-readable evidence and reproduction instructions
are indexed in `submission/final_artifacts.json`.
