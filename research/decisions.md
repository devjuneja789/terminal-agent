# Research decisions

## Submission candidate frozen; final matched evaluation (2026-11-15)

- The user approved the submission-day plan: one final fixed 300-update/20-epoch
  support-only fit, then candidate freeze and final matched evaluation. Same
  15 admitted targets, fresh base, joint objective, optimizer, seed, quantization
  and AMP settings as EXP-023; the linear schedule stretches with the horizon.
  No objective sweep or new data. Training and reload pass all checks.
- EXP-024's exact-context audit passes all lineage/token/weight/control checks.
  Joint-300 reaches 15/15 correct choices, 4/4 finish choices, 6/15 exact actions
  and 15/15 schema-valid outputs, against joint-60's 8/15, 0/4, 3/15 and 14/15.
  It wins the preregistered support-only tuple and is frozen at 07:14:40 UTC,
  artifact `1eed40612640c78d88d9fe0cac36e06b1148d1fd`. Do not overwrite the
  freeze or train again after held-out inspection. Correct choices are not
  executable task success; argument and answer mismatches persist.
- EXP-025 reruns both untouched base and selected adapter on all ten permanent
  held-out tasks with identical native-stop/NF4/v5 settings. The historical
  FP16/v1 baseline remains historical. Verify task bytes against Git; no
  held-out task/validator changes. Stage no training data and generate no rewards.
  Job `6af95c1129dd05ff0368583f` completed as run `20261115T071710Z`.
  Matched contract passes are 1/10 versus 2/10, but the additional B050 pass
  contains factual errors. Both models complete 0/5 repairs and make no file
  changes. Keep the nominal metric and trace counterexample together; no
  reliable end-to-end gain is claimed and no benchmark label is changed.
- AuditKIT is a real pinned core integration at commit
  `9d4e21bac240e8108a22c2c137d179d6070868d5`, installed version 1.1.12.
  Import actual AgentTune EventLogs into its recorded AgentEvalRunner; recompute
  the contract from recorded validator and tool evidence. This is offline
  auditing, not new model/tool execution or a judge API. Emitted-action structural
  scores have a different denominator from rejected-attempt harness metrics.
- Prepare public code, attributed adapter cards, explicit data/log payloads,
  provenance and a recorded walkthrough before owner release review. No base
  weights are redistributed; no live inference is implied by the demo.

## Joint decision/completion experiment completed (2026-11-15)

- The user authorizes necessary project commands and HF job submissions.
  Scope each run with a fixed data/config/budget and private artifact destination;
  no rerun of an already populated adapter repository or unbounded training.
- EXP-023 keeps the same 15 admitted targets, fresh base and 60 updates at
  scale 256. Optimize 0.75 first native tool-name-token CE plus 0.25 normalized
  weighted full-completion CE; all argument/code/closing coefficients positive.
  No text, schema, prompts or model initialization changed.
- All real gradient, effective coefficient, update and reload gates pass. The
  complete exact-context comparison recovers all 45 records and reproduces all
  base controls. New adapter greedy outputs and teacher top-1 arrays match
  EXP-022 on all 15 contexts: choices 8/15, writes 2/3, finish 0/4, exact 3/15.
- Full-action NLL improves 0.785→0.754 without changing decisions. Hub LFS file
  hashes differ and the loader pins the new artifact, so do not confuse stable
  argmax outputs with accidental checkpoint reuse. No task-success gain.
- Primary choice and schema goals fail. Stop without agent/held-out evaluation
  or further automatic training/data admission. A future fixed fit-budget
  diagnostic may clarify insufficient optimization; current evidence does not
  establish a capacity limit or justify more unbounded objective changes.

## Tool-choice/finish isolation completed (2026-11-15)

- EXP-022 retains the four admitted references/15 targets, fresh base and
  60-update/four-epoch budget. Optimize only native tool_name value tokens,
  including finish. Preserve full reviewed templates and label-presence audits,
  but explicitly record zero effective loss on arguments/code/native closings.
  Retained labels are not equivalent to effective optimizer supervision.
- The scale-8192 startup aborts at the first skipped AMP update, with zero
  optimizer updates and no trained artifact. Retain raw failure and v1 profile.
  Correct the numerical startup in v2 to fixed scale 256 with an all-15-batch
  checkpointed-gradient probe before optimizer creation; no scale search,
  objective/LR/data/initialization change or training-budget extension.
- Dummy-logit isolation and real gradient probes pass. Corrected training
  completes all 60 updates, zero skips, changed/reloaded FP32 adapters. Tool
  loss falls 1.408→0.518 across epochs; do not compare this numerically with
  full-completion loss. The AMP change is a disclosed numerical-control caveat.
- Repeat EXP-021's exact inference procedure on base/new adapter. All 15 base
  token outputs and teacher top-1 arrays reproduce, and all 45 records/label/
  effective-weight/native-stop audits pass. Valid tool choices improve 6→8/15,
  write choices 0→2/3, exact actions remain 3/15; finish 0/4 and read 0/2.
  Primary 15/15 choice criterion is not met. Schema validity falls to 14/15.
- All three write-target raw outputs attempt write_file; count B026 malformed
  JSON as invalid. B021 edits the wrong constant; B041 retains functions and
  changes default 15 but drops header/import. No writes/tests execute here,
  so static file proposals are not benchmark passes or completed repairs.
- Stop without a full-agent replay, longer run or new target admission. Read/
  finish transitions and edit arguments remain unresolved; concentrated
  tool-name loss alone is not a demonstrated agent improvement.

## Exact corrective-context predictions completed (2026-11-14)

- EXP-021 stages only the four admitted references, 15 exact targets and the
  completed EXP-020 training audit. Reproduce every prompt/target token ID,
  label position/count and weight before model inference. No trainer, optimizer,
  backward, terminal tools, benchmark snapshots or held-out tasks enter the run.
- Thirty base/adapter predictions share pinned NF4/FP16/eager greedy inference,
  seed 42, 256-token limit and native stops. All 15 context reproductions and
  30 native-stop audits pass; all 45 records recover without another GPU run.
- Adapter exact actions improve 1/15→3/15 and tool-name matches 1/15→6/15.
  All three exact predictions are primary test commands. All six shell targets
  choose shell, but three repeat the primary test instead of the upper-bound
  regression test. Read targets 0/2, write targets 0/3, finish targets 0/4.
- All predictions are strict-schema-valid. Wrong action transitions persist
  even with exact reviewed contexts: repeated reads before writes, test calls
  instead of complete reads, and extra commands instead of finishing. No write
  or final-answer predictions exist to credit as file preservation/paraphrases.
- Teacher token accuracy rises 86.69%→89.64% and NLL falls 0.811→0.528,
  but gold-context token scores do not substitute for correct greedy actions.
  Missing runtime coverage cannot be the sole explanation of full-loop failure.
  This does not prove a model-capacity limit or a specific optimization cause.
- Stop after the diagnostic; no new targets admitted and no training follows.
  Next proposed experiment: isolate critical tool-choice/finish learning on the
  same approved targets under a fixed versioned plan before more agent replays.

## Bounded corrective SFT and support replay completed (2026-11-14)

- EXP-020 starts a fresh LoRA from the pinned base on exactly 15 admitted
  corrective continuations: 60 updates/four epochs, unchanged weighted loss,
  no old-gold merge, audit-only targets or supervision of archived prefixes.
  A separate cloud staging path verifies admission hashes and both passed
  preflight/runtime gates; actual trainer masks/closings/weights are rechecked.
- AgentTune transcript export must round-trip exactly. Its factory selects
  explicit continuation indices, rather than extracting every assistant turn.
  Both audit-only reports and all creation-time review artifacts stay unchanged.
- All 60 updates complete, zero skipped steps, finite loss, FP32 trainable
  adapters, maximum length 2490/4096, successful private save/reload. Epoch
  mean weighted loss falls 0.807→0.521. This is pipeline/learning-loss evidence,
  not an agent-success claim.
- Compare fresh base and the new adapter under the validated native-stop
  profile; only replay version/adapter identity differ from EXP-019. Both
  score 0/4, no edits. New adapter schema validity is 83.33%, versus the old
  weighted adapter's 100%; no cumulative gain is claimed. Thirty generations
  use the union; 29 close normally and one reaches the unchanged token cap.
- Preserve false success claims and malformed calls. Discarded trailing shell
  proposals are not executed edits. B046's intentional failing test remains
  valid evidence; partial causal explanation does not pass its fixed contract.
- Zero task-opening targets and zero exact training-prompt visits suggest a
  coverage gap, without proving causality. Stop after the fixed experiment.
  Next: conditional predictions on the exact reviewed contexts before changing
  data, initialization or training budget. No held-out access or output admission.

## Versioned native-stop support replay completed (2026-11-14)

- EXP-019 runs the existing base/weighted adapter on the same four support tasks
  with native union `[3,6,261001]` in both arms. All historical config fields,
  task contracts, snapshots, validators and initial prompts match EXP-012;
  only the version and opt-in stop/telemetry fields differ.
- All 38 generations use the union and end once on END_RESPONSE, with zero
  cap hits. IDs 3/6 remain configured but were not emitted. Base generated
  tokens fall 3840→783, adapter 965→946. Treat this as runtime efficiency,
  not a learned planning gain. Both models remain 0/4; destructive writes,
  missing edits and false success claims remain recorded without admission.
- The live CLI omitted final log chunks despite successful completion. Recover
  complete logs from the same job; never substitute six records for eight or
  spend on another inference job just to recover logs. Independent audit
  recomputes metrics and hashes all eight records plus source/log provenance.
- Save a validated opt-in native-stopping profile and runtime gate. Preserve
  the inactive EXP-018 candidate and all frozen legacy configs/results as
  historical artifacts. Future comparisons must explicitly version and use
  the same stopping profile in both models. No global runtime switch here.
- Stop before training. Next is bounded corrective-data training with the 15
  admitted continuation targets and existing masking/weight checks. No new
  owner review is required for this runtime gate; no held-out tasks accessed.

## Pinned corrective label/stopping preflight completed (2026-11-14)

- EXP-018 verifies all 15 admitted targets using the actual pinned TRL collator.
  All prompt turns are masked, targets decode exactly, both native closing
  tokens are supervised at weight 1, and no truncation occurs (max 2490/4096).
  The three metadata-free full-read weight annotations now work using conservative
  EOF evidence; old gold spans remain unchanged. No data text is rewritten.
- Official pinned model/generation JSON use EOS 3, tokenizer turn EOS is 6,
  and chat END_RESPONSE is 261001. Keep these distinctions explicit. Legacy
  generation stops only at 6 and ignores the emitted response boundary.
- In twelve matched generations, all six legacy controls reproduce and the
  response-aware tokens match each legacy prefix through its first closing.
  Base generated tokens fall 768→121, cap hits 3→0, decoded actions unchanged.
  This fixes a generation-efficiency issue in the diagnostic, not task behavior.
- Historical speed/token comparisons are affected by this stopping interaction;
  do not attribute all their differences to learned planning. Preserve old runs
  and validate a new consistent runtime on both models before new comparisons.
- Save an inactive candidate preserving upstream EOS 3 plus closings 6/261001.
  Only [261001,6] was directly tested here; the full union requires a versioned
  live support replay. Frozen inference settings remain unchanged.
- All 27 audit records recovered; no optimizer, gradients, training, local model
  inference or held-out access. Next: versioned native-stop support comparison,
  then bounded training on the audited corrective export. See EXP-018.

## Corrective data admission completed (2026-11-14)

- Apply all six owner decisions without rewriting their reasons. Preserve the
  original receipt, raw candidates and pending sidecar backup. Admission is a
  separate step authorized by the user's subsequent request to proceed.
- Admit only four task-completing references; both approved factual failure
  reports remain permanently excluded. Revalidate all four transcripts against
  frozen support snapshots and task contracts before exporting.
- Export 15 unique continuation-only targets, zero exact duplicates or old-gold
  overlaps. Do not merge old data or train archived assistant prefix turns.
  Generic all-assistant extraction is inappropriate for these transcripts.
- All four executions and 15 structural boundary checks pass; expected nonzero
  B046 test exits remain valid evidence. No new model success, inference,
  training, benchmark change or held-out access occurs. See EXP-017.
- Token-level audit remains pending because local package versions differ from
  the pinned cloud runtime. A cached tokenizer inspection also identifies the
  previously unexplained token 261001 as END_RESPONSE; the runner stops only on
  tokenizer EOS 6, END_OF_TURN. Check both closing tokens and official generation
  settings in the next pinned preflight before allocating further training.
  Do not silently change frozen evaluation settings or historical results.

## Corrective support batch prepared for owner review (2026-11-14)

- EXP-016 branches from actual weighted replay prefixes on B021/B026/B041/B046.
  Re-execute those prefix tools to reproduce state, preserving original messages
  and hash-bound provenance. No new model inference, training or held-out access.
- Three repair continuations observe complete source before full-file writes,
  change only the requested behavior, and pass named plus upper-bound checks.
  The explanation continuation uses the actual intentionally failing test and
  source evidence without making changes or claiming the test passed.
- Two additional truthful reports follow destructive writes/test failures.
  They remain failed tasks and audit-only, with zero proposed training targets.
  Correct factual reporting is not a positive repair result.
- The first four candidates propose 15 actions. Archive prefix assistant turns
  are context only and cannot become completions. Explicit continuation indices
  and completion-only row inspection enforce that distinction.
- All six records are pending human review and training-ineligible. Old approved
  data, owner reviews and training launchers stay unchanged. Refuse overwriting
  the review batch and bind decisions to exact source-record hashes. No approved
  export or automatic training admission is produced.
- These are six branches on four tasks, not six new independent tasks or evidence
  of model improvement. Owner review, duplicate handling and a bounded training
  gate remain before any new GPU experiment. See EXP-016 and the review packet.

## Observation-format and metadata comparison completed (2026-11-14)

- EXP-015 corrects the unverified serialization hypothesis: all twelve reviewed
  observations already use the runtime's JSON header, indentation and escaping.
  TinyAyaPolicy serializes the same payload byte-for-byte identically. Do not
  invent a formatting bug or regenerate approved references just for whitespace.
- The bounded-result adapter clears metadata on every tool result. Four reviewed
  reads and three writes contain metadata; five shell results already have none.
  Save the static audit and source hashes separately from inference results.
- In matched full-source live histories, restore only truthful read metadata
  (path and line_count). All twelve predictions are recovered and the three
  runtime adapter outputs reproduce EXP-014 exactly. No gold answers are supplied.
- Metadata leaves B021's malformed action and B026's repeated read unchanged.
  B041 adds contradictory default assignments (15 then 10), turning its named
  test pass into a failure. Both models have zero strict preserved/exact gold
  writes. Preserve the regression; do not patch output or alter validators.
- Metadata omission is real but is not a demonstrated fix for these failures.
  Keep the production normalizer, frozen tasks and reviewed data unchanged.
  Stop formatting-only probes; next consider reviewed corrective support
  demonstrations in actual runtime histories before allocating more training.
- No training, held-out access, new agent benchmark score or automatically approved
  target is produced. Last full replay remains EXP-012's 0/4. See EXP-015.

## Matched live-source observation intervention completed (2026-11-14)

- EXP-014 supplies complete source in the actual archived failing histories,
  changing only the last observation's stdout. Original conversations are exact;
  prior read arguments stay unchanged, so augmented observations are explicitly
  counterfactual. Both base and weighted adapter receive identical paired inputs.
- All twelve predictions were recovered. The three original adapter outputs
  reproduce EXP-012 exactly. B026's inputs and outputs remain identical across
  conditions in both arms, validating its role as an identity control.
- Complete source improves B041 from a broken snippet to a write whose named
  timeout test passes, with both functions preserved. It still removes the
  unused os import and module docstring: a scope defect, not an observed failure
  of the named functional repair. Report both facts without erasing the gain.
- B021's augmented output is malformed JSON and is never executed. B026 still
  reads instead of writing. Full source alone is insufficient for reliable edits.
  Do not weaken validation or automatically fix the model's JSON.
- Keep conditional named-validator passes separate from strict file preservation
  and complete agent success. No new agent task-success rate is claimed; the last
  actual paired replay remains 0/4. No generated output becomes training data.
- Before more training, consider a separately versioned observation-serialization
  comparison under the same live histories. Full-source gold-history success
  did not transfer exactly, and remaining context differences are not isolated.
  No extra training, held-out access or benchmark edit occurred. See EXP-014.

## Complete-source preservation diagnostic completed (2026-11-14)

- EXP-013 reused the saved weighted adapter and exact EXP-011 source prefixes,
  input IDs, labels, quantization and decoding. All 48 boundary/prediction records
  were recovered; no optimizer, training or held-out access occurred.
- With complete approved source, B041's entire action/file matches the gold edit
  and its named test passes. B021 retains surrounding code and passes its named
  lower-bound test, but incorrectly removes the upper cap. B026 still reads
  instead of writing. Do not describe these as three complete repairs.
- Run existing B021 upper-bound test as an explicitly post hoc diagnostic;
  it fails 101 != 100. Preserve the frozen named validator and report both
  outcomes. No task or benchmark file is changed after seeing this result.
- Keep file preservation, semantic edit correctness and source gathering separate.
  Live B021/B041 wrote after partial reads and deleted module contents; approved
  full-source histories produce preserved files. Other history details differ,
  so source length alone has not been isolated causally. B026 already had full
  source in the live run and remains an action-choice problem.
- Exact free action matches improve from EXP-011's 5/16 to 6/16, adding the B041
  write to five shell commands. This is conditioned prediction evidence; the
  last actual agent replay remains 0/4. No generated target is owner-approved.
- Next consider a controlled source-observation intervention on support tasks,
  uniformly across base/adapter under a separately versioned replay. Do not
  launch more training or change the frozen benchmark based on this diagnostic.

## Weighted completion objective completed (2026-11-14)

- EXP-012 keeps the four approved references/16 targets, seed, 64 updates,
  runtime and replay controls fixed. Tool names and changed-code tokens receive
  weight 8, JSON closure weight 4, other completion tokens/EOS weight 1;
  context remains masked. Use maximum overlap and summed-weight normalization.
- Derive edit spans from the preceding approved full-file read and map exact
  escaped JSON offsets to supervised tokens. Audit all 16 targets and reject
  ambiguous/truncated labels. Synthetic unit weights reproduce ordinary CE
  values and gradients; no model inference runs locally.
- The first weighted job overflowed FP16 gradients and was cancelled. Preserve
  its failed-attempt record/log. Initialize AMP at 8192 on rerun, recording the
  original 65536 state and unchanged growth/backoff rules; fail immediately on
  skipped updates. This numerical difference from EXP-010 is disclosed.
- Corrected training completed all 64 updates/four epochs without skips.
  The replay recovered all eight trajectories: both arms 0/4. Weighted calls
  were 19/19 schema-valid, with three executed writes on B021/B041; all writes
  replaced required module contents and failed validation. B026 falsely claimed
  a fix without a write; B046 omitted the actual ignored-environment cause.
- Treat this as narrow protocol/tool-choice progress, not a working repair
  model. All four adapter episodes still needed repeat recovery. Preserve
  model-origin final answers but admit none as training data. No held-out
  access or evaluation changes occurred. See EXP-012 and its independent audit.
- Before more training, inspect whether the weighted model can produce complete
  preserved-file edits under approved full-file histories versus its live
  partial reads. Do not automatically increase epochs or sweep weights.

## Per-action label diagnostic completed (2026-11-14)

- EXP-011 verified all 16 training prefixes, decoded completion labels and EOS
  targets against generation prompts. All counts reproduced. The prepared-row
  shuffle concern caused no source-ID mismatches in this run; do not report
  it as a discovered audit bug.
- With approved reference histories, the 64-step adapter reproduced 5/16
  exact actions versus 1/16 for base. All five were test-running shell calls.
  It selected write_file on 0/3 intended edit actions, preferring shell at the
  first tool-name token even with 96.75–98.67% per-action token accuracy.
- Preserve all 16 boundary and 32 prediction records. These conditioned
  predictions are diagnostic evidence, not new task-success or training data.
- Next hypothesis: emphasize critical tool-name, changed-code and JSON-closure decisions within
  full completion supervision and track per-action matches. No extra training
  or data expansion was launched as part of the diagnostic.

## Fixed memorization diagnostic completed (2026-11-14)

- EXP-010 used exactly 64 fresh-base QLoRA updates and four epochs on the same
  four approved references/16 targets, with no skipped updates. Save all 64
  training logs and the exact adapter upload commit. No extra data or held-out
  tasks entered this experiment.
- The paired replay remained 0/4 for base and adapter with zero edits. The
  adapter reduced invalid calls (16.7% to 7.7%) and extra-action/prose violations
  (6 to 0). Report that limited protocol effect without claiming task learning.
- Preserve the raw token-length value of 2 as an instrumentation bug. Audit
  mask counts yield a maximum actual batch length of 2345. Future preflight
  counts input_ids, not BatchEncoding fields; regression coverage was added.
- Stop after the fixed experiment. Diagnose per-action predictions and label
  boundaries before another training escalation; aggregate loss and valid JSON
  are insufficient to demonstrate terminal-task completion.

## Paired support replay completed (2026-11-13)

- EXP-009 compared base and the eight-step adapter under identical NF4
  inference on B021/B026/B041/B046. Both scored 0/4, made no file changes,
  and had 83.3% schema validity. The result is training-example replay, not
  held-out performance. Preserve all eight raw trajectories and log provenance.
- Check effective chat templates rather than serialization containers. The
  initial job's preflight rejected equivalent dict/string representations; the
  corrected retry compared selected text, vocabularies and special token IDs.
- Keep support observations in JSON for both arms. The missing v5 formatting
  branch was corrected before inference and did not alter the frozen baseline.
- Eight optimizer steps covered only 0.5 epoch of 16 action targets. A future
  fixed multi-epoch diagnostic should establish whether verified action
  sequences can be learned before expanding data or running held-out models.
  Do not label weight changes, lower loss, or an unverified finish as task success.

## SFT pipeline milestone (2026-11-13)

- HF run `20261113T220511Z` completed eight QLoRA steps on the four approved
  references, with changed adapter weights and a successful reload. The dataset
  hash matches the local 16-target export; no held-out tasks were accessed.
- Preserve the raw result and source-log provenance under
  `research/results/tiny-aya-sft-smoke/`. The generated support probe has an
  invalid JSON escape and was not executed. Do not label this task success or
  interpret aggregate training loss as improved agent performance.
- Next diagnose behavior with a paired base/adapter replay on the approved
  support tasks using the same NF4 inference settings. Treat it as training-set
  reproduction and keep it distinct from the earlier FP16 held-out baseline.

# Current decisions and status (2026-11-15)

This record summarizes the active Terminal-SLM hackathon implementation and
supersedes the earlier prototype-specific environment and model notes above.
The active model path is Tiny Aya Global on Hugging Face GPU Jobs through
AgentTune. Local model inference is not part of this submission.

## Tool and workspace boundary

- Keep the five tools (`shell`, `read_file`, `write_file`, `grep`, `git`)
  workspace-scoped with bounded time/output, filtered environments, command
  logs, path checks and explicit destructive-operation guards.
- Use temporary task workspaces and independent validators. These controls are
  application-level guardrails, not an operating-system sandbox.
- Preserve complete trajectories, raw failures, timestamps and model/harness
  attribution. Never turn a harness-authored termination message into a model
  training target.

## Model, training and evaluation boundary

- Use `CohereLabs/tiny-aya-global` for the hackathon comparison. AgentTune is
  the model/action and training integration; the local harness is not the
  inference path for reported results.
- Freeze a 50-task benchmark at 40 support and 10 held-out tasks. Held-out IDs,
  outcomes, patches and rewards remain outside training and support selection.
- Admit only reviewed support targets. Keep truthful-failure demonstrations
  audit-only, and distinguish completed edits from factual-answer scoring.
- Treat executable validators as the task-success authority and AuditKIT as a
  recorded-episode structural audit. Neither training loss nor tool-choice
  accuracy alone demonstrates task completion.
- Keep support-only diagnostics separate from the frozen matched held-out
  comparison, with the model/configuration and limitations stated for each.

## Completed evidence and current interpretation

- Tiny Aya baseline: 0/10 frozen held-out task contracts (EXP-004).
- EXP-025 matched held-out comparison: base 1/10, selected adapter 2/10;
  both 0/5 on code repair/update tasks. The adapter's extra contract pass had
  an inaccurate explanation, so no reliable end-to-end gain is claimed.
- EXP-027 later support-only diagnostic: task-start adapter 2/4 versus base
  0/4 on four known support tasks. Two exact reviewed edits passed named and
  regression tests; two tasks failed. This is not held-out evidence and did
  not alter the EXP-025 candidate or metrics.
- Training scripts verify source provenance, masks, gradients, updates, and
  adapter reload. The saved training loss is not a task-success measurement.

Old local-model prototype trajectories have been removed from the current
working tree. Earlier commits retain repository history; this cleanup does not
rewrite Git history. The active support data and Tiny Aya records preserve their
own review, split and provenance boundaries.

## Submission boundary

No public model/data publication or live deployment is claimed. Before any
public release, review Tiny Aya's base-model terms, confirm the project's own
source/data licensing, publish only approved artifacts, and verify links. No
additional training or inference is recommended before submission.
