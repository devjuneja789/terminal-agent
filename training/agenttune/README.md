# AgentTune and Hugging Face GPU workflow

## Joint first-decision and full-action experiment

EXP-023: `bash scripts/run_tiny_aya_joint_sft.sh` submits a new private-adapter
run on the same 15 approved targets. Its fixed loss is 0.75 CE on the first
native tool_name token plus 0.25 normalized EXP-020 weighted full-completion
CE. Every argument/code/closing token therefore retains positive loss. The
fresh-base 60-update plan, AMP scale 256 and all-batch gradient preflight remain
fixed. A new GPU self-check verifies prompt masking, nonzero completion gradients
and emphasis on the first decision. No reviewed action text is rewritten.

After training, use
`TERMINAL_SLM_CONTEXT_TRAINING_RESULT=<result-path> bash scripts/run_tiny_aya_joint_contexts.sh`.
Audit with `python3 scripts/analyze_corrective_contexts.py --prefix <prefix> --training-result <result-path>`.
This compares the exact same 15 contexts without running generated tools.
No full-agent or held-out evaluation follows automatically.

Completed training `20261115T062137Z` and diagnostic `20261115T063016Z` pass
all pipeline audits. Valid choices remain 8/15, finish 0/4, exact actions 3/15;
all adapter outputs match EXP-022 despite distinct Hub file hashes and lower
full-action NLL. No decoded-behavior improvement or agent replay is claimed.
See EXP-023 for the fixed criteria and complete evidence.

## Tool-choice-only isolation

EXP-022 keeps the same 15 approved corrective targets and fresh-base 60-update
plan, but computes loss only on native tool_name value tokens, including finish.
Run `bash scripts/run_tiny_aya_tool_choice_sft.sh`. No reviewed text or template
labels are rewritten; all nonchoice loss coefficients, including argument/code
and closing tokens, are zero. Label-presence audits describe retained templates;
the effective-supervision annotations separately identify the optimized tokens.
A CUDA dummy-logit check verifies gradient isolation before training updates.

Use the completed result in
`TERMINAL_SLM_CONTEXT_TRAINING_RESULT=<result-path> bash scripts/run_tiny_aya_tool_choice_contexts.sh`.
This repeats the exact-context base/adapter procedure without terminal tools.
Audit using `python3 scripts/analyze_corrective_contexts.py --prefix <prefix> --training-result <result-path>`.
The primary memorization criterion is 15/15 valid correct tool choices, with
write/finish counts reported explicitly. Stop after the fixed run and diagnostic;
no full-agent replay follows automatically, even if tool-choice accuracy improves.

EXP-022 records an initial zero-update overflow at AMP scale 8192. The preserved
v1 profile and failed-startup record are separate from the completed v2 run.
V2 uses fixed scale 256 and a finite-gradient probe on all 15 batches before
optimizer creation. Corrected training `20261115T055053Z` completes 60 updates,
zero skips, with validated save/reload. Diagnostic `20261115T055808Z` reproduces
all base controls: valid choices 8/15, write 2/3, finish 0/4, exact actions 3/15.
The learning criterion is not met. All 45 records are retained; no agent replay
follows. Arguments/closures were not optimized and must not be treated as fixed.

## Exact reviewed corrective-context diagnostic

`bash scripts/run_tiny_aya_corrective_contexts.sh` runs EXP-021: 15 exact
admitted prompts on both the pinned base and EXP-020 corrective adapter.
No trainer, optimizer, backward pass or terminal tool execution occurs.
The bundle contains no repository snapshots or held-out tasks. It reproduces
training prompt/target IDs, label positions and weights against the completed
training annotations before inference. Both models use the validated native
stop union with the same 256-token greedy inference limit.

The extractor requires all 15 contexts and 30 predictions. Run
`python3 scripts/analyze_corrective_contexts.py --prefix <result-prefix>` for
independent lineage, label, strict-action, teacher-token and stopping checks.
Report greedy tool choice/action equality separately from teacher-forced token
accuracy and legitimate finish paraphrases. Conditional predictions are not
completed repository tasks and are never automatically admitted for training.

Completed run `20261114T224529Z`: all 45 records and exact training-token/label/
weight checks pass. Base/adapter exact actions 1/15 and 3/15; tool choice 1/15
and 6/15. The adapter still selects no write_file on three write targets and
no finish on four finish targets. All schemas are valid and all 30 generations
stop on END_RESPONSE. See EXP-021: critical action learning remains insufficient
even with exact reviewed context; no new training or agent replay follows.

## Bounded admitted-corrective training

EXP-020 uses `bash scripts/run_tiny_aya_corrective_sft.sh`: a fresh LoRA from
the pinned base, 60 updates/four epochs over the exact 15 admitted corrective
continuations, with the existing weighted loss and FP32-adapter precision fix.
The separate preparer requires the immutable admission/export hashes, EXP-018
label gate and EXP-019 native-stop runtime gate. It stages no benchmark files,
old gold targets or audit-only failure reports. The trainer verifies AgentTune's
full transcript round trip but labels only explicitly admitted continuation
actions. Actual TRL labels, native closings and weights are checked again
before optimizer creation; any skipped update aborts the run.

After recovering a complete `.result.json`, use
`TERMINAL_SLM_REPLAY_TRAINING_RESULT=<result-path> bash scripts/run_tiny_aya_corrective_replay.sh`.
The replay preparer rejects an old, partial or differently configured adapter.
It changes only the EXP-019 version and adapter identity, keeping native stops
`[3,6,261001]` on base and adapter. The eight support trajectories remain
ineligible for automatic training admission. Independently audit with
`python3 scripts/analyze_corrective_training.py --training-result <result-path> --prefix <replay-prefix>`.
No automatic longer run or held-out evaluation follows this experiment.

Completed training `20261114T222003Z`: all 60 updates/four epochs pass; epoch
mean loss 0.807→0.521, all 15 actual label checks pass, save/reload succeeds.
Completed replay `20261114T222638Z`: both models remain 0/4, no edits. Native
stopping remains valid (29 closing tokens, one cap hit across 30 generations).
See EXP-020 for full failure review. Next is an exact-reviewed-context action
check before any further training, not an automatic longer run.

## Versioned native-stop support replay

`bash scripts/run_tiny_aya_native_stop_replay.sh` runs EXP-019: base and the
existing EXP-012 weighted adapter on B021/B026/B041/B046, with native stop IDs
`[3,6,261001]` in both arms. It derives the old replay configuration and checks
that only its version and opt-in stopping/telemetry fields differ. The temporary
bundle contains no corrective demonstrations or held-out tasks. No training runs.

Every generation logs its IDs, active stopping policy and cap/closing status.
After extracting all eight complete trajectories, run
`python3 scripts/analyze_native_stop_replay.py --prefix <result-prefix>`.
This independently checks configuration and initial-prompt parity, token
accounting, closing/cap behavior, and metrics against raw records. A passed
runtime gate does not establish a task-success gain or admit training examples.
Legacy replay configurations retain their original tokenizer-only stopping.

Completed run `20261114T220329Z`: all eight trajectories recovered; all 38
generations stop at END_RESPONSE without cap hits. Base generated tokens
3840→783, adapter 965→946; both remain 0/4. The `.runtime-gate.json` records
independent checks and provenance. The live CLI lost final chunks; fetch the
same job logs and extract again rather than rerunning inference. This gate
permits planning the next bounded corrective experiment, not automatic training.

## Pinned corrective label and stopping preflight

`bash scripts/run_tiny_aya_corrective_preflight.sh` stages only hash-verified
admitted rows and three saved support histories. On HF T4 it constructs the
actual TRL SFTTrainer/collator without an optimizer or train call, audits all
15 label/weight boundaries, then compares legacy and response-aware stopping
on both base and adapter. Its extractor requires all 27 records. Run
`python3 scripts/analyze_corrective_preflight.py --prefix <result-prefix>` for
independent boundary, control-reproduction and generation-prefix checks.

Completed run `20261114T213811Z`: all 15 label checks pass, maximum length 2490,
and both native closings are supervised. Base generated tokens fall 768→121
and cap hits 3→0, with unchanged decoded actions. Upstream EOS is 3, turn EOS
6, response end 261001. The inactive union proposal in
`configs/tiny_aya_stopping_candidate.json` records the historical proposal.
EXP-019 validates the full union separately; the validated opt-in profile is
`configs/tiny_aya_native_stopping_v1.json`. No frozen settings or historical
results are changed. See EXP-018/019.

## Corrective data admission

EXP-017 applies the owner's six review decisions and separately admits only four
corrective demonstrations. The dedicated export is
`data/sft/tiny_aya_corrective_v1.jsonl`: 15 unique continuation-only targets.
Both audit-only reports are permanently excluded. See `data/corrective/README.md`
for immutable source/receipt artifacts and applied/admitted status files.

`training.agenttune.admit_corrective_support` checks hash-bound approvals,
re-executes reviewed actions in fresh snapshots, exports explicit continuation
indices, checks duplicates/conflicts, and preserves the old pending sidecar.
Do not use generic `action_rows()` on the admitted full transcripts; that would
include archived assistant prefix turns. Existing cloud training launchers have
not been redirected to this dataset. Pinned token/label and stopping preflight
must pass before further training. No new training has run.

## Corrective support review candidates

`python3 -m training.agenttune.build_corrective_support --output <new-path.jsonl>`
replays actual support prefixes and executes assistant-authored corrective
continuations in disposable workspaces. It produces candidates, a hash-bound
pending review sidecar, readable packet and manifest; it refuses overwrites.
The prepared batch is `data/corrective/support_corrective.v1.*`.

Four task-completing candidates propose 15 continuation-only action targets.
Two truthful failed-task reports are audit-only and excluded. Archived assistant
turns are never completion targets. All proposed rows are training-ineligible;
owner review and a separate admission/export step remain required. Existing
SFT launchers continue using the old approved data. No training or inference
runs in this builder. See EXP-016 for execution checks and owner review details.

## Observation-format and metadata parity

`bash scripts/run_tiny_aya_observation_metadata.sh` runs EXP-015. Its static audit
checks actual reviewed observations against the runtime serializer and bounded
result adapter. JSON serialization already matches; only metadata is removed
for the reviewed read/write results. The GPU comparison changes only read
metadata within the exact EXP-014 full-source histories. Both models receive
the same six inputs; no gold targets or training entrypoints are bundled.

The existing analyzer detects this diagnostic profile:
`python3 scripts/analyze_observation_intervention.py --prefix <result-prefix>
--run-validators` (place both flags on the same command). It validates metadata-only
pairs and applies only eligible writes to disposable snapshots. Results are
under `research/results/tiny-aya-observation-metadata/`.

Completed run `20261114T202246Z`: runtime controls reproduce exactly. Restored
metadata changes neither B021's invalid JSON nor B026's repeated read, and
regresses B041 from a named-test pass to failure through duplicate assignments.
No training, held-out evaluation or production-normalizer change ran. See EXP-015.

## Matched live-history source intervention

`bash scripts/run_tiny_aya_observation_intervention.sh` stages only the three
saved failing conversations and their original/full-source observation pairs.
The inference-only HF T4 job generates twelve next actions across base/adapter;
it bundles no gold answers, benchmark records or training entrypoint. Only the
source stdout field changes. B026 is an identity control; original adapter
outputs are checked against the archive. Runtime and decoding remain fixed.

`scripts/analyze_observation_intervention.py --prefix <result-prefix>` checks
the matched inputs and scores preservation. Add `--run-validators` to apply
only eligible file contents in temporary snapshots and run the existing named
validators plus the separately reported B021 upper-bound check. It executes no
model-generated shell commands and reports no complete agent-success metric.

Completed run `20261114T182159Z` recovered all twelve predictions and reproduced
all three original adapter outputs. Complete source makes B041's named test
pass, but the write drops its import/docstring. B021 emits invalid JSON and B026
does not write. No training or held-out inference ran. See EXP-014 and
`research/results/tiny-aya-observation-intervention/` for full evidence.

## Complete-source preservation diagnostic

`bash scripts/run_tiny_aya_source_preservation.sh` reuses the inference-only
action diagnostic with the pinned EXP-012 weighted adapter. It retains the
EXP-011 prompts, token sequences, greedy decoding and 256-token limit, scoring
all 16 approved action prefixes on fresh base and weighted adapter. No optimizer
or training call runs. The primary edit prefixes already contain full original
source; compare their results with the archived EXP-011 unweighted predictions
and EXP-012 live decisions. Histories differ beyond source length, so this is
not a causal source-length ablation.

After extraction, `scripts/analyze_source_preservation.py` accepts
`--diagnostic-prefix` and `--output-prefix`. It checks data/artifact lineage and
unchanged token sequences, then audits whole-module AST preservation and diffs.
With `--run-validators`, only schema-valid writes to the expected path are
applied to disposable snapshots and tested with the existing named validators.
Generated shell commands are never executed by this analyzer. These conditional
checks are not agent task-success scores. See EXP-013.

Completed run `20261114T175228Z`: all 48 records recovered, unchanged token
sequences verified. Under complete approved histories, B041 reproduces the
exact gold write and passes its test; B021 preserves surrounding code and
passes the named lower-bound test but fails an explicitly separate existing
upper-bound test; B026 still selects read_file. These are conditional predictions,
not a new live success rate. No further training or held-out inference ran.

## Fixed weighted-loss diagnostic

`bash scripts/run_tiny_aya_weighted_sft.sh` runs EXP-012 on HF T4 GPUs: a fresh
64-step adapter followed by the existing paired four-task replay. It uses the
same approved text, pinned runtime, seed and inference settings as EXP-010.
Only the completion objective changes: tool names and changed code 8x, JSON
closure 4x, all other completion tokens/EOS 1x, masked context zero. Overlaps
take the maximum, normalized by the sum of active weights. Full-file read/write
diffs determine changed spans; exact tokenizer offsets determine token weights.
The saved `loss-weight-audit.jsonl` records every mapped span and target weight.
Unit weights are tested against ordinary CE values and gradients. Weighted and
ordinary completion loss are logged separately. Artifacts stay private, and
training/inference bundles exclude held-out tasks. See EXP-012 for results.

Completed EXP-012: initial FP16 overflow required a documented startup-scale
correction (65536 to 8192), then all 64 updates completed without skips. Both
replay arms scored 0/4. The weighted adapter executed three writes with 19/19
schema-valid terminal calls, but overwrote required module code and repaired
no task. Full results and audits are in `research/results/tiny-aya-weighted-sft/`
and `research/results/tiny-aya-weighted-replay/`. Do not train from these outputs.

## Per-action label and prediction inspection

`bash scripts/run_tiny_aya_action_diagnostic.sh` stages only the four approved
references and the validated EXP-010 artifact identity. It reconstructs TRL
labels without an optimizer/train call, then loads fresh base/adapter models
for teacher-forced scoring and greedy next-action predictions. No tools run.
It inspects all 16 targets, matches shuffled processed rows by full contents,
and compares training label prefixes to actual inference prompts. Results
include critical tool-token ranks/probabilities and strict action matches.
The HF T4 job has a 30-minute cap. Sixteen label records and 32 predictions
must be recovered completely under `research/results/tiny-aya-action-diagnostic/`.
See EXP-011. These gold-history predictions are not executable task results.

Completed run `20261114T152447Z` recovered all 48 diagnostic records. All 16
label boundaries and EOS labels match; the possible shuffled-source-ID issue
was not realized (zero mismatches). Base versus adapter exact next-action
matches were 1/16 versus 5/16, and schema-valid generations 7/16 versus 15/16.
The adapter's five exact matches were all test-running shell actions; it never
selected write_file on the three edit targets. See EXP-011 for tool-token ranks
and the proposed tool-choice-focused objective. No training occurred.

## Fixed four-epoch memorization diagnostic

`bash scripts/run_tiny_aya_memorization.sh` executes the fixed EXP-010 plan:
64 fresh-base QLoRA steps (four epochs over the same 16 approved targets), then
a separate paired GPU replay with the exact EXP-009 controls. It creates a
new private adapter repository, records every training log entry in bounded
chunks, and pins the upload commit for inference. The replay launches only
after the expected steps/epochs finish without skipped updates and the
approved-data lineage is verified. Training and inference bundles remain
separate. Job caps are 25 minutes for training and 30 for replay.

The eight-step profile remains in `configs/tiny_aya_sft_smoke.json`; the new
profile is `configs/tiny_aya_memorization.json`. Outputs go to
`research/results/tiny-aya-memorization/` and `tiny-aya-memorization-replay/`.
Use only train-support tasks. This diagnoses memorization, not generalization.

EXP-010 completed: training `20261114T145152Z` reached 64 steps/four epochs
with no skipped updates; paired replay `20261114T145802Z` recovered all eight
records. Both arms passed 0/4 tasks and made no edits. The adapter reduced
invalid calls from 16.7% to 7.7% and extra-action/prose violations from 6 to 0.
Full training history and the trace audit are saved with the experiment.
The recorded token-length preflight value of 2 was an encoding-field-count
bug; audit counts establish a 2345-token maximum batch. Future preflight code
now measures input_ids explicitly. Original raw artifacts are preserved.

## Paired support replay

`bash scripts/run_tiny_aya_paired_replay.sh` compares the untouched base and
the EXP-008 adapter on B021, B026, B041 and B046, with identical NF4 settings
and the support harness. The adapter commit is pinned. Each of eight episodes
gets a fresh snapshot copy; no reference transcripts or held-out task records
are bundled. The existing evaluator captures complete traces and validators.
The job uses t4-small with a 30-minute limit. Outputs are recovered losslessly
from keyed chunks into `research/results/tiny-aya-paired-replay/`.
See EXP-009 for the procedure fixed before inference. These are training tasks,
so the comparison is a reproduction diagnostic, not held-out performance.

Completed run `20261113T223053Z` recovered all eight episodes: base 0/4,
adapter 0/4, with zero file edits and identical schema validity (83.3%). The
first setup-only attempt stopped on an overly strict template-container check;
the retry verified identical effective template text, vocabulary and special
token IDs. Full results and trace audit are recorded in EXP-009. No longer
training job was started.

## Approved-reference SFT smoke (prepared 2026-11-13)

The four owner-approved v1 references convert to 16 next-action targets with
`python3 -m training.agenttune.prepare_sft`. Conversion validates the exact
reviewed record hashes. Only assistant action completions receive loss;
observations remain context. See `data/sft/tiny_aya_sft_smoke.manifest.json`.

The AgentTune 1.1.3 source at commit
`d4db4c12d52342227c4113a70183d0f64aa2f345` was inspected to wire
`Project.train` to a TRL SFTTrainer factory without rewriting Cohere JSON calls.
The GPU smoke performs eight QLoRA steps, saves and reloads the adapter, and
records a short support probe. V2 completed on HF T4 as `20261113T220511Z`:
adapter updates and reload passed, but the single generated probe contained
invalid JSON and was not executed. See EXP-008 and the saved structured result.

The first cloud attempt failed on BF16 gradient unscaling with FP16 AMP. V2
casts only trainable LoRA parameters to FP32 after TRL initializes them; the
NF4 base and FP16 compute are preserved. The same private repository may be
reused if it contains only `.gitattributes` from the failed attempt. Public or
populated destinations are rejected. Retry with the same launcher command;
the local bundle is rebuilt automatically. The EOS-alignment warning is not
the cause of that failure. See EXP-008 for source inspection and job ID.

Choose a NEW private Hugging Face model repository ID; the job creates it.
After confirming that destination, run from an activated local venv:

```sh
TERMINAL_SLM_ADAPTER_REPO=devjuneja789/CHOOSE-NEW-NAME bash scripts/run_tiny_aya_sft_smoke.sh
```

The T4 Job has a 25-minute limit and displays live hardware pricing. All model
loading, training, and probe inference happen on the HF GPU. The bundle excludes
all benchmark tasks, rejected references, and pending v2 replacements. Adapter,
runtime metadata and metrics are uploaded privately under the run ID. Local
logs are in `research/results/tiny-aya-sft-smoke/`. A successful pipeline smoke
does not demonstrate better coding behavior; inspect EXP-008 before expanding.

Phase 1 uses a Hugging Face GPU Job for Tiny Aya Global inference. The local
machine only stages the code bundle and submits the job; model weights are
downloaded inside the job. The bundle includes the workspace-scoped `tools/`
code and train-support task B002. It excludes `.venv`, model weights,
credentials, raw results, and held-out task records.

The smoke runs one real AgentTune `Project` episode using `ReActStrategy` and
`DictToolHarness`. Tiny Aya uses AgentTune's documented Cohere fallback format
and receives tool results as user turns. The existing `Toolbox` executes tools
inside a temporary copy of the miniapp snapshot. The smoke validates the final
response, a successful required `grep` call, and the selected task's validation command.
It is not training or a formal benchmark run.

## Phase 2 frozen held-out evaluation setup

The formal baseline config is `configs/tiny_aya_eval.json`. It pins Tiny Aya's
model revision, cloud package versions, decoding, prompt, schema hash, and the
ten permanent held-out task IDs. The evaluator is
`training/agenttune/evaluate_heldout.py`; it uses the existing benchmark
snapshot-copy/validation runner and workspace-scoped `Toolbox`. It requires
CUDA and refuses to start if the model revision, benchmark IDs, runtime pins,
prompt hash, or schemas differ.

The held-out job bundle is built by
`training/agenttune/prepare_hf_eval_bundle.py`. It carries only the held-out
task file, a manifest containing only held-out IDs, the required miniapp
snapshot, and runtime code. It explicitly excludes `train_support.jsonl`.
Keep this separation for the baseline; future rollout/training preparation
must use a separate support-only bundle.

To reproduce the Phase 3 baseline, check the live price and launch from the
repository root:

```sh
hf auth whoami
hf jobs hardware
./scripts/run_tiny_aya_baseline.sh
```

The job has a 40-minute hard timeout. The launcher preserves the complete HF
Job output and extracts `research/results/tiny-aya-baseline/<run-id>.jsonl`,
`<run-id>.metadata.json`, and `<run-id>.summary.json`. Phase 3 is complete; the
first run and separate trace-recovery run are recorded in EXP-004. Both scored
0/10; the recovery run preserves all ten trajectories. Do not merge the two
runs' records or metrics. Phase 4 is support-only rollout/reward validation;
do not use held-out tasks or traces as training examples or reward targets.

## Phase 4 support-only rollout and reward pilot

The prepared pilot is ten preselected training-support IDs, one per category:
`B001, B006, B011, B016, B021, B026, B031, B036, B041, B046`. The bundle
builder exposes only the `train-support` manifest and task file. It generates a
support config that inherits the baseline model revision, prompt, schemas,
decoding, and package pins, while carrying only the support pilot IDs. It does
not bundle `heldout_eval.jsonl`, held-out IDs, or the held-out eval config.

The runner executes each task in a fresh workspace. It records the pristine
validator result, final validator result, changed task files (excluding tool
logs/caches), complete AgentTune and Terminal-SLM trajectories, and a
deterministic reward audit. A code-change task can receive task-success reward
only if its pristine validator fails, the agent changes a task file, and the
benchmark contract/validator pass afterward. Invalid actions, failed tool
execution, repeated actions, and excess calls have explicit small costs.
Navigation/discovery/explanation tasks receive a null reward pending factual
human review because their current keyword-based task contracts cannot prove
the final answer is correct. Every candidate has `training_eligible: false`
until reviewed; these are not yet SFT/GRPO inputs.

The pilot completed as run `20261109T070739Z`. The human sidecar is
`research/results/tiny-aya-support-rollouts/20261109T070739Z.review.jsonl`.
Apply it to a corrected rescore while preserving the raw trajectories:

```sh
python -m training.agenttune.apply_support_reviews \
  --run research/results/tiny-aya-support-rollouts/20261109T070739Z.jsonl \
  --review research/results/tiny-aya-support-rollouts/20261109T070739Z.review.jsonl \
  --output research/results/tiny-aya-support-rollouts/20261109T070739Z.reviewed.jsonl
```

The corrected rescore has zero contract passes and zero training-eligible
candidates. See EXP-005 before authorizing any data conversion.

V4 has completed. Its raw job summary reported 1/10, but trace review found
that B006 read only a signature; rescoring against the current task contract
reports 0/10. Seven repeated-action records contain a harness-generated final
response. The reviewed copy excludes those records from training targets. The
owner confirmed all ten rejections; all records remain training-target-
ineligible.

V5 implements one finish-only model retry after a repeated tool action. The
duplicate is not executed. If the retry is malformed or proposes another tool,
the harness stops with an empty final response; it does not manufacture answer
text. The focused support check on B016, B046, and B041 completed as
`20261112T200940Z`. It reported two contract passes, but trace review shows
B016 did not report whether the function reads the environment variable,
B046's explanation overstates the test evidence, and B041 made no edit. No
`write_file` call occurred and no candidate is training-eligible. See EXP-006.
The saved artifacts are under `research/results/tiny-aya-support-rollouts/`.

To launch a future support-only run after a concrete behavior or data change:

```sh
source .venv/bin/activate
hf auth login
TERMINAL_SLM_SUPPORT_TASK_IDS=B016,B046,B041 ./scripts/run_tiny_aya_support_rollouts.sh
```

After review, rescore a generated support run and apply its sidecar without
modifying raw trajectories with `python -m training.agenttune.apply_support_reviews
--run RUN.jsonl --review RUN.review.jsonl --output RUN.reviewed.jsonl`. It
validates run/task IDs and snapshot hashes, recalculates reward fields, and
surfaces audit disagreements. It never makes candidates training eligible by
default.

For v4, loop-guard fallback text is not a model completion. The reviewed output
marks those records with `training_target_excluded: true` and reason
`harness_generated_loop_guard_response`; do not convert either that field or
the harness-authored finish action in the trajectory into an assistant target.
The v5 policy uses an empty harness stop after an unsuccessful finish-only
retry, and candidate records are explicitly `training_target_eligible: false`.

## Support contract audit and reference traces

EXP-007 corrected the support success conditions for B016, B021, B026, B041,
and B046. Source evidence now binds to the expected `shop/config.py` path;
change tasks require an actual `write_file` and successful test output, and
B026 requires both the observed initial failure and passing rerun. The task
split and held-out records are unchanged.

Six canonical, assistant-authored traces are stored in
`data/sft/gold_support.jsonl`. Rebuild them with
`python3 -m training.agenttune.build_gold_support`. The builder makes no model
calls; it executes actions and validators in throwaway snapshot copies. Every
record is pending maintainer review and has `training_eligible: false`. This
is not a trainer input yet, and no new cloud rollout or training job has been
started. See `research/experiments/EXP-007-support-gold-contract-audit.md`.

## Run the Phase 1 smoke

From the repository root in WSL:

```sh
source .venv/bin/activate
python -m pip install --upgrade huggingface_hub
hf auth login
hf auth whoami
hf jobs hardware
./scripts/run_hf_gpu_smoke.sh
```

`hf auth login` prompts for the token; do not put it in this repository or send
it in chat. The logged-in account must have access to `CohereLabs/tiny-aya-global`.
The launcher forwards the cached token to the Job as the `HF_TOKEN` secret.

The launcher selects `t4-small`, sets a 20-minute job timeout, and prints the
current hardware table first. Hugging Face currently lists T4-small at
$0.40/hour and bills by the minute, so a full 20-minute run is about $0.14 at
that listed rate. Confirm the live price shown by `hf jobs hardware`; the job
timeout caps the smoke run. No exposed endpoint is created.

The Job prints one line beginning `PHASE1_SMOKE_RESULT=` with the actual
trajectory, model output, validation result, GPU, package versions, and outcome.
Keep the Job ID and that record. If the smoke fails, report the full failure
record and Job logs; do not interpret a failed run as a model pass.

The Docker command uses `--` before `bash /workspace/training/agenttune/run_smoke_job.sh`
to pass the command through to the container, as required by the Jobs CLI when
separating options from the container command.

## Current execution status

The GPU Jobs confirm Tiny Aya loads and generates. B006 exposed weak argument
recovery: the model repeatedly used invalid `read_file` line numbers. On B002,
it found the path and emitted a correct plain-text answer after proposing
another read; the old adapter discarded that answer and kept reading. The
runner preserves raw completions, feeds back only the selected action, and now
converts a plain-text tail after an observed tool step into AgentTune's
`finish` action. The B002 GPU smoke passed with this adaptation. See
`research/experiments/EXP-002-tiny-aya-phase1-smoke.md` and raw records under
`research/results/`. Hardware and package versions were not present in the
successful result; collect them for the formal baseline. No local GPU inference
or training was run.
