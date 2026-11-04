# Terminal-SLM implementation plan: Lexsi Track 3

**Status:** Phases 0 and 1 are complete. The user confirmed hackathon
participation, Tiny Aya Global access, and HF GPU credits. As reported by the
user, the balance is $19.87, current-period usage is $0.14, and the current
period ends Dec 1, 2026. A Hugging Face GPU Job completed the B002 route-table
discovery through Tiny Aya,
AgentTune, `grep`, observation, and final answer. B006's function-purpose task
still failed due to invalid argument recovery; these failures remain documented
as limitations. Raw attempts are recorded in EXP-002 and `research/results/`.
No model was downloaded locally, no training was started, and the benchmark
was not changed.
**Reviewed:** 2026-11-02
**Proposed track:** Track 3 — Agentic Fine-tuning

## Final decisions

- **Track:** Track 3 — Agentic Fine-tuning.
- **Base model:** `CohereLabs/tiny-aya-global` (Tiny Aya Global), used as both
  the untouched baseline and the starting checkpoint for the tuned adapter.
  It is in the smallest listed chat-model tier at 3.35B parameters, with an
  8K input context, making it the most practical of the
  seven choices for a short, credit-limited tool-use experiment. The reason for
  choosing it is size and instruction-following suitability, not multilingual
  terminal-task needs. Verify Hub access and accept/review its license before
  downloading it.
- **Training/evaluation pipeline:** AgentTune for tool rollouts and
  post-training; AuditKIT for paired agent evaluation.
- **Project strategy:** adapt the existing Terminal-SLM repository. Do not
  start over. Keep the sandboxed tools, task snapshots, fixed splits, and Qwen
  baseline as prior project work. Qwen is not the hackathon submission model or
  the comparison baseline for the Tiny Aya experiment.

This satisfies the published Cohere Labs model requirement while preserving
the work that is directly reusable. The prior Qwen3 baseline (1/10 held-out
success) motivates tool-reliability work, but it is not a before/after result
for Tiny Aya.

The task matches the project's existing weakness: the untouched Qwen3 baseline
produced schema-valid tool calls, yet completed only 1/10 held-out tasks. Its
trajectories show path guessing, poor recovery after tool errors, and failure to
make or validate requested changes. Those are direct targets of Track 3's
reliable tool calling and efficient trajectory problem statements. The Qwen3
run remains useful historical evidence, while Tiny Aya Global becomes the
eligible model for the hackathon's own paired experiment.

## Track requirements and fit

The hackathon describes Track 3 as agent fine-tuning for accurate, well-formed,
correctly ordered, and efficient tool calls. Its stack points to AgentTune for
post-training and AuditKIT for evaluation. The event rules require a dense,
open-weights Cohere Labs model, before/after results on more than one metric,
public deployment, and Hugging Face hosting for produced artifacts. The
two-repository minimum has an explicit Track 3 exception, but using AgentTune
and AuditKIT together makes the submission's pipeline clear.

| Current project | Proposed hackathon version |
|---|---|
| Qwen3-1.7B was the planned target | Use Tiny Aya Global for the hackathon baseline and tuned model; preserve Qwen3 artifacts as prior research only. |
| Custom model loop and Qwen `<tool_call>` parser | Keep the terminal workspace/tools, but add an AgentTune integration for model rollouts, training, and trajectory format. Use the documented Cohere fallback/tool-result handling; do not assume Tiny Aya's tokenizer supports the Qwen native tool-call template. |
| Local benchmark evaluator and metrics | Keep the deterministic task runner as an executable task environment; export or adapt AgentTune trajectories into AuditKIT agent episodes and compare baseline vs. tuned model. |
| QLoRA SFT, then optional DPO/GRPO | Make one small AgentTune agentic fine-tuning experiment the core contribution. Pilot its GRPO/LoRA path on support tasks; if the reward signal or budget makes that impractical, use AgentTune's supported trajectory-distillation/SFT path on reviewed support-only demonstrations and document the change. |
| Local GTX 1060 Max-Q, 6 GB VRAM | Run all Tiny Aya model inference, training rollouts, and benchmark evaluation in a Hugging Face GPU environment. Use the local machine for editing and CPU-only tests; do not load the model or run model generation locally. |

## Proposed research question

> Does AgentTune post-training improve Tiny Aya's ability to complete
> workspace-scoped terminal tasks, with fewer failed or redundant tool calls,
> compared with the same untouched Tiny Aya model?

Primary metric: executable task success rate. Supporting metrics: required-tool
selection accuracy, argument/schema validity, invalid-call rate, tool calls and
model turns per task, tokens per task, latency, and failure categories. Use the
same tasks, validators, workspace restrictions, decoding settings, and limits
for the base and tuned Tiny Aya evaluations. Report every metric and raw
trajectory; do not claim improvement from a metric that rewards tool use
without verified task completion.

## Proposed pipeline

```text
train-support tasks only
        ↓
Terminal-SLM workspace-safe tools + AgentTune rollout/training
        ↓
Tiny Aya base → small LoRA adapter
        ↓
AgentTune trajectories → AuditKIT agent evaluation
        ↓
paired before/after metrics + public demo and Hugging Face artifacts
```

### Model and tool-call compatibility

Tiny Aya Global is a gated Hugging Face model governed by the Cohere Labs
CC-BY-NC license and acceptable-use policy. Confirm account access and comply
with those terms before downloading, training, or publishing an adapter.

The current Terminal-SLM parser specifically expects Qwen's
`<tool_call>{...}</tool_call>` format. AgentTune's current documentation says
the Cohere hackathon models do not provide a tokenizer chat template that
renders tool schemas and tool-result turns; AgentTune provides a fallback that
places tool descriptions in the prompt and returns tool results as user turns.
The implementation should use and test AgentTune's supported path and its
parser/format, or explicitly configure a supported tool-capable template. Do
not silently pass Qwen tool schemas and assume Tiny Aya will honor them. Keep
the existing parser for the Qwen historical path if useful; isolate any new
adapter behind the model/rollout boundary.

Expose the existing five tools (`shell`, `read_file`, `write_file`, `grep`,
`git`) as AgentTune tools that call the existing `Toolbox` inside one isolated
task workspace. Preserve path restrictions, command guardrails, timeouts,
output limits, and complete trajectory logging. Avoid switching to a broader
filesystem or shell environment just to simplify the training integration.

### Data and training boundaries

- Use `train-support` only to create or select training rollouts and examples.
- Keep the permanent held-out IDs in `evals/benchmark/manifest.json` out of
  every training, preference, reward-model, and hyperparameter-selection path.
- Preserve the current Qwen3 held-out results and `EXP-001` as historical
  records. They are not the before measurement for Tiny Aya.
- First evaluate untouched Tiny Aya on the fixed held-out split. Then fine-tune
  from support data and evaluate the adapter once using the same frozen
  procedure. Do not tune prompts or rewards against individual held-out tasks.
- Keep a human-reviewed record of each training example's source task ID,
  rollout, expected tool/action, validation result, and any edits. Do not treat
  model-generated answers as correct without running the task validator.
- Reward verified task state and correctness first. Use small, explicit costs
  for redundant calls or excessive steps. A final answer string or merely
  calling the required tool must not earn task-success reward by itself.

The current benchmark has only 50 tasks, including 40 support and 10 held-out.
Treat the evaluation as a small benchmark result; show task-level outcomes and
raw traces alongside aggregates. Do not modify this benchmark after inspecting
results. If a real benchmark defect is found, record it, create a new benchmark
version and frozen split, then rerun both model conditions under that version.

## Feasibility and decision gates

There are about 12 days between this review and the published Nov 14 submission
deadline. Registration closes Nov 5. Treat these as gates, not assumptions:

1. **Phase 0 — complete:** user confirmed participation, Tiny Aya Global access,
   and the active $20 credit. The precise coupon expiry is not a blocker because
   the project must be submitted by Nov 14; finish all credit-funded runs before
   that deadline.
2. **Cloud runtime:** use a Hugging Face GPU-backed runtime for every model
   generation, tool rollout, baseline evaluation, and post-training run. Run
   the benchmark and isolated task workspaces inside that cloud environment so
   model inference does not depend on the local GPU or a remote inference API
   from WSL. Pin a separate cloud environment. AgentTune currently documents
   Python 3.12+, PyTorch 2.11, and Transformers 5.15+; do not install its
   training stack into the project `.venv` without a compatibility check.
3. **Budget/VRAM pilot:** confirm a GPU instance that can load the model and
   run the selected PEFT training configuration. Check the live price and
   expected runtime before starting. The $20 credit may constrain the number of
   runs; stop if the pilot cannot finish within the credit balance. Do not
   assume the local 6 GB GPU can train this 3.35B model.
4. **Training go/no-go:** run a short support-only pilot and inspect actual
   losses, rollouts, task rewards, and memory use. Continue only if tool actions
   are being executed and validated rather than merely formatted.
5. **Evaluation and submission:** compare untouched and tuned Tiny Aya on the
   same fixed held-out tasks with at least the primary and supporting metrics.
   Publish reproducible code, adapter/model artifacts, logs, and permitted
   datasets on Hugging Face, plus a public project/demo link and a concise
   write-up by the deadline.

## Work to keep, defer, and replace

### Keep

- The isolated workspace and restricted `Toolbox` implementation.
- The current miniapp snapshot, task contracts, fixed split manifest, and
  evaluator, unless an independently confirmed defect requires a versioned
  benchmark correction.
- Existing real Qwen3 traces, the Qwen3 baseline report, and manual smoke
  results as historical evidence.

### Change for the submission

- Add AgentTune as the training/rollout lifecycle and AuditKIT as the
  before/after agent-evaluation stage.
- Add a tested Tiny Aya/AgentTune compatibility path rather than reusing the
  Qwen-only chat-template assumption.
- Create a separately pinned cloud training environment and a small, explicit
  training configuration.
- Add a Hugging Face artifact and public demo/presentation plan.
- Update README positioning after the integration works; avoid claiming the
  model has improved until the paired evaluation is complete.

### Defer

- DPO, a second RL method, GGUF/llama.cpp deployment, and larger-model
  comparisons. They do not directly improve the minimum Track 3 demonstration
  and compete for the limited time and credits.
- Expanding to many task categories or building a new generic agent framework.
  Prioritize a reliable small end-to-end comparison and transparent artifacts.

## Phasewise execution plan

The hackathon page lists registration through **Nov 5, 2026** and submission by
end of day **Nov 14, 2026**. This plan is intentionally narrow enough for that
window. Dates below are targets; do not skip an exit gate to meet a date.

| Phase / target | Work | Exit gate and artifacts |
|---|---|---|
| **0. Eligibility and access — complete** | User confirmed hackathon participation and Tiny Aya Global access. Credit status reported 2026-11-02: $19.87 balance, $0.14 current-period usage, period ends Dec 1, 2026. | Complete. Hackathon submission deadline is Nov 14; record live Job prices and costs for each run. |
| **1. Hugging Face GPU runtime and smoke — complete** | Tiny Aya loaded and completed B002 in a HF GPU Job. The adapter converted its post-observation plain-text final into AgentTune's `finish` action. B006 failures are retained as limitations. | Basic end-to-end cloud smoke passed. Hardware/package versions were not returned; record them in the formal baseline. Local GPU was not used. |
| **2. Integrate and freeze evaluation — complete Nov 4** | Added a pinned Tiny Aya/AgentTune runtime config, frozen prompt/tool-schema hashes, a held-out evaluator, and a cloud runner. Reuse the five workspace-scoped tools, benchmark snapshots, runner, held-out IDs, and validators. Keep the Qwen native parser isolated from Tiny Aya/AgentTune. | Frozen config and held-out-only bundle preparation are in place. No task definitions changed; no training data is bundled. No baseline result exists yet. |
| **3. Tiny Aya base baseline — complete Nov 4** | The untouched model ran all ten frozen held-out tasks on a Hugging Face T4. The first run's aggregate is saved; a same-config recovery run reproduced the metrics and captured all ten complete task trajectories. | 0/10 success on both runs. Full recovery artifacts are recorded separately in EXP-004. Proceed to support-only rollout and reward validation; do not use held-out data for training. |
| **4. Support rollouts and reward check — Nov 5–7** | Use only `train-support` tasks. Adapt task validators into deterministic AgentTune reward functions: task-state/validation success is primary; tool/schema correctness and a modest cost for failed/redundant steps are diagnostics/secondary terms. Inspect actual rollout and reward examples manually. | Every training task has source ID and validation evidence. Reward sanity checks reject a wrong final answer, no-op edit, or merely calling a required tool. No held-out prompt, trajectory, outcome, or reward enters training. |
| **5. One bounded post-training pilot — Nov 7–10** | Run a short AgentTune GRPO + PEFT/LoRA experiment on the support split, with explicit step/runtime caps and spend monitoring. Save checkpoints, training logs, configs, and rollouts. Do not launch multiple sweeps. | Training finishes within the verified budget and produces a loadable adapter; sample traces show the policy is learning executable actions. If GRPO is unstable or unaffordable, use AgentTune's documented SFT/distillation path only with reviewed, validated support-only demonstrations, and record this as the actual method. If neither trains, report the blocker honestly; do not fake an improvement. |
| **6. Paired AuditKIT evaluation — Nov 10–12** | Evaluate the untouched Tiny Aya base and tuned adapter with identical held-out tasks, prompt, tools, limits, and validators. Use AuditKIT's agent-evaluation path and preserve Terminal-SLM raw trajectories. Report task success plus multiple supporting metrics: tool accuracy/schema validity, tool steps/loops, tokens, latency, and failure types. | Paired before/after report and per-task traces exist. No post-hoc task/prompt changes. Report regressions and uncertainty; do not claim success if only a proxy metric improves. |
| **7. Public packaging and submission — Nov 12–14** | Publish code/demo, model adapter, training configuration, permitted dataset/provenance, logs, and evaluation artifacts to Hugging Face. Write the project article/readme with thesis, model/license, method, metrics, compute cost, limitations, and reproduction steps. Check that no credentials, local paths, or disallowed artifacts are published. | Public repository/demo link and complete submission by Nov 14. Preserve exact artifact revisions and final evaluation run IDs. |

## Start-from-scratch decision

**Do not start the project over.** Reuse the sandboxed tool layer, CLI,
benchmark snapshots, fixed split, evaluation outputs, and Qwen3 records. The
local Qwen checkpoint was removed; keep only the source code and historical
Qwen result records. Add a Tiny Aya/AgentTune path and run it entirely in the
Hugging Face GPU environment. The local machine is for editing and CPU-only
checks. If the current benchmark has a confirmed defect, version and freeze a
corrected benchmark before using it; do not quietly patch task definitions
after seeing Tiny Aya results.

## Sources reviewed (2026-11-02)

- [Hackathon tracks, schedule, credits, and rules](https://hackathon.lexsi.ai/)
- [AgentTune repository and Tiny Aya integration notes](https://github.com/Lexsi-Labs/AgentTune)
- [AuditKIT repository and agent evaluation](https://github.com/Lexsi-Labs/AuditKIT)
- [CohereLabs/tiny-aya-global model card](https://huggingface.co/CohereLabs/tiny-aya-global)
- [CohereLabs/tiny-aya-base model card](https://huggingface.co/CohereLabs/tiny-aya-base)
