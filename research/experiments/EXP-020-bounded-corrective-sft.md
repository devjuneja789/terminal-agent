# EXP-020: Bounded corrective SFT and native-stop support comparison

## Fixed plan before execution

Hypothesis: training on the four admitted continuations from actual failing
conversations can improve complete-file repairs, action choice and factual
explanations in the full agent loop. This is a support-data learning diagnostic,
not held-out generalization.

Use exactly the 15 exported targets (SHA-256 canonical rows
`03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b`),
four admitted examples C-B021-read-complete-v1, C-B026-choose-write-v1,
C-B041-preserve-file-v1 and C-B046-explain-evidence-v1. No old gold merge,
new data generation, excluded factual-failure reports or archived prefix
supervision. Keep review/admission/label/stopping gate artifacts immutable.

Start a fresh LoRA from the pinned Tiny Aya base, not the old weighted adapter.
Keep EXP-012's NF4/FP16 compute, FP32 trainable adapters, q_proj/v_proj rank 8,
alpha 16, dropout 0, learning rate 1e-4, linear schedule, no warmup, seed 42,
batch/accumulation 1, checkpointing, 4096 tokens, AMP initial scale 8192 and
8x tool-choice/8x changed-code/4x JSON-closure weighted completion loss.
Adjust update count to **60 = four epochs over 15 targets** (old set had 16).
Supervise both native assistant closings at weight 1. Recheck actual TRL
collator labels and weights before any optimizer update. Abort on skipped
updates, nonfinite loss, data/gate drift, truncation or changed boundaries.

Use AgentTune's Project.train with full exact transcript round-trip validation;
the trainer factory selects only admitted continuation indices. Save/reload
the adapter to a new private HF repository. T4-small, 25-minute job cap;
one run only, no adaptive extension or automatic retry with altered settings.

After a successful audited training run, run the full agent loop once with
fresh base and new adapter on unchanged B021/B026/B041/B046, seed 42,
NF4/FP16/eager and native stops `[3,6,261001]` in both arms. Derive from
EXP-019, changing only version and adapter artifact identity. Preserve all
prompts, schemas, limits, tasks, snapshots, validators and logging. T4-small,
30-minute job cap. Recover eight full trajectories and all token-stop audits;
if streaming truncates, fetch completed logs instead of rerunning inference.

Report training loss by epoch, update/mask/weight/reload checks, immutable
artifact revision, task success, writes/file preservation, truthful tests,
schema/selection/steps/tokens/latency and failure categories. Compare to the
same native-stop base and EXP-019's old weighted adapter, with factual trace
review. A narrow named-test pass is insufficient evidence of file preservation;
check remaining source and existing upper-bound behavior where relevant.
Do not admit generated outputs automatically. Stop after this experiment;
no held-out evaluation, dataset expansion or further training follows.

## Status

Complete: training and the full eight-episode support replay are audited.

## Training result

HF Job [6af8de3d23985fb9cd829fdc](https://huggingface.co/jobs/devjuneja789/6af8de3d23985fb9cd829fdc),
run `20261114T222003Z`. All 60 optimizer updates completed, exactly four epochs,
zero skipped steps, all 15 actual label-mask checks passed, three weighted edit
targets, maximum sequence length 2490/4096. All trainable LoRA parameters are
FP32 (1,916,928); frozen base NF4 compute remains FP16. Adapter weights changed
and reloaded successfully. T4 peak allocated CUDA memory 8,232,136,192 bytes.
Training runtime 211.4392 seconds, overall mean weighted training loss 0.62945.

| Epoch | Mean weighted loss | Targets |
|---|---:|---:|
| 1 | 0.80716 | 15 |
| 2 | 0.63394 | 15 |
| 3 | 0.55548 | 15 |
| 4 | 0.52122 | 15 |

Private adapter repository `devjuneja789/terminal-slm-tiny-aya-corrective-60-v1`,
immutable revision `3c151e4280bab3fe27c25a53992a3cc17fc785fc`, subfolder
`20261114T222003Z/adapter`. The real reload probe selected a test-running shell
action; this is a reload/inference check, not evidence that the repair works.

Local artifacts: `research/results/tiny-aya-corrective-sft/20261114T222003Z`
`.result.json`, `.training-history.jsonl`, `.provenance.json`, `.training-gate.json`.
Private HF artifacts also retain adapter weights, tokenizer, training history
and weight annotations. Raw source/review/admission/export artifacts unchanged.

Local checks: 142 tests run, 139 pass and three optional Torch checks skipped;
five new corrective training/bundle tests pass. Shell syntax, Python compilation
and `git diff --check` pass. No local model training/inference or held-out access.

The comparison is HF Job
[6af8dfc523985fb9cd82a0e8](https://huggingface.co/jobs/devjuneja789/6af8dfc523985fb9cd82a0e8).
Replay run `20261114T222638Z`. Both Jobs independently confirmed COMPLETED:
349 training-job running seconds and 286 replay-job running seconds. At the
listed T4-small rate $0.40/hour, their combined running duration corresponds
to about $0.071; this is an estimate, not a billing receipt.

## Paired replay results

| Metric | Base, native stopping | New corrective adapter | Previous weighted adapter, EXP-019 |
|---|---:|---:|---:|
| Task contract passes | 0/4 | 0/4 | 0/4 |
| Tool selection accuracy | 50.0% | 50.0% | 63.64% |
| Argument/schema validity | 83.33% | 83.33% | 100% |
| Invalid call rate | 16.67% | 16.67% | 0% |
| Mean tool steps/task | 2.00 | 2.00 | 3.75 |
| Mean model turns/task | 3.75 | 3.75 | 5.75 |
| Generated tokens, total | 783 | 1093 | 946 |
| Mean total prompt+generated tokens/task | 6807.00 | 6811.25 | 11893.50 |
| Mean agent latency/task, seconds | 14.50 | 23.47 | 23.29 |
| Executed write_file calls | 0 | 0 | 3 |

Failure categories in each current arm: two invalid_tool_call and two
validation_failure. All eight raw trajectories, prompts, model outputs,
selected/blocked actions, executed tools, observations, finals, validators
and generation-token audits are saved in
`research/results/tiny-aya-corrective-replay/20261114T222638Z.jsonl`.
Companion `.metadata.json`, `.summary.json`, `.md`, `.analysis.json`,
`.context-coverage.json`, `.provenance.json` and `.job-inspection.txt` retain
configuration, independent recomputation, source/log/artifact hashes and
completed-job status. All eight records recovered from the original live log;
no GPU retry or log recovery was needed.

Every one of the 30 generations used `[3,6,261001]`. Twenty-nine end on
END_RESPONSE, one adapter B026 generation reaches the unchanged 256-token
cap. No generation continues past a recognized closing token. The cap hit
is a lengthy multi-action proposal, not evidence that EOS was ignored.
The audit verifies native policy, token accounting and summary recomputation,
and exact EXP-019 configuration equivalence except version/adapter identity.

## Factual trajectory review

| Task | New adapter behavior | Outcome |
|---|---|---|
| B021 | Proposes a sed command with an invalid JSON escape; parser rejects it. | No command executes, no edit, original lower-bound test still fails. |
| B026 | Runs the failing test, searches with shell/grep, repeats the test and falsely reports a completed change/passing test. | No edit; `10 != 25` remains. |
| B041 | Searches for the instruction text as a literal pattern, observes the failing test, then claims grep confirmed 15 despite no matching evidence. | No edit; `10 != 15` remains. |
| B046 | Runs the expected failing test, searches/reads source, also attempts line 0 arguments. Final mentions hard-coded 10 versus 25 and the environment variable. | Partly causal explanation, but invalid arguments and missing frozen explanation terms; no contract pass. |

No files change in any episode, and neither arm executes write_file. B026's
raw first completion proposes additional shell actions, including echo-based
overwrites, after its first test call; those unobserved proposals are discarded
by the existing one-action policy and **do not execute**. Do not describe them
as actual edits. The harness correctly blocks repeated calls and rejects
malformed JSON/arguments. Do not relax the parser or benchmark to promote these
failures. B046's intentional nonzero test exit remains valid evidence, not an
executor failure; its partial factual improvement is not a task-success gain.

The fresh corrective adapter does not retain the previous weighted adapter's
schema/write behavior. This experiment changed its training data while starting
from the base; it did not continue the old adapter. Both source identity and
initialization are explicit, and no cumulative improvement is claimed.

## Conclusion and next step

**Training pipeline passes; this bounded corrective training does not improve
end-to-end task success.** Loss falls, masks are correct, adapters update/reload,
and stopping works, but the agent still fails before producing verified repairs.
These facts do not establish that the model learned the desired actions on
their exact supervised contexts; that requires a separate conditional check.

The 15 targets contain zero task-opening prompts (two-message conversations).
No exact reviewed training prompt is visited in this replay. This suggests a
coverage/activation gap worth investigating; it is not proof of the cause,
because model/tool trajectories and temporary traceback paths also differ.
Keep the original admitted targets/reviews unchanged. Do not automatically
extend training, admit these outputs, change held-out tasks or claim a positive
post-training result. Next recommended bounded diagnostic: predict actions on
the exact 15 reviewed prompts to distinguish conditional learning failure from
failure to reach the corrected states, then decide whether opening/anchor
demonstrations or a different initialization are justified. No further inference,
training, new data admission or held-out evaluation follows automatically.
