# Terminal-SLM: final matched evaluation

Actual run: `20261115T071710Z`. All 20 raw records pass the independent freeze, contract and aggregate checks.

## Question and method

Can a small Cohere model learn reliable terminal actions from reviewed executable support demonstrations using AgentTune?
The terminal harness executes five workspace-scoped tools; AgentTune supplies the native Cohere JSON-action loop and training integration. AuditKIT evaluates actual recorded episodes with an independent contract oracle and structural scorers.

Selected adapter: `devjuneja789/terminal-slm-tiny-aya-joint-300-v1` at `1eed40612640c78d88d9fe0cac36e06b1148d1fd`, subfolder `20261115T064937Z/adapter`.
Frozen before this run at `2026-11-15T07:14:40.667391+00:00` from the preregistered support-only rule. A later, separately reported support-only diagnostic (EXP-027) trained a different adapter; it did not change this frozen candidate, use held-out information for training, or evaluate the new adapter on held-out tasks.

Base: `CohereLabs/tiny-aya-global` at `af89d219b53ed9b13b8a4645f9c8028973510324`.
Both models use identical NF4/double quantization, FP16 compute, eager attention, seed 42, greedy decoding, native stop union `[3,6,261001]`, prompts, schemas, limits and permanent held-out tasks.

## Results

**Factual caveat:** the additional adapter contract pass, B050, contains an inaccurate explanation. Neither model completes a code repair (0/5 each). The nominal score change does not establish reliable task-success improvement. See [the trace review](TRACE-REVIEW.md).

| Metric | Untouched base | Selected adapter |
|---|---:|---:|
| Contract successes | 1/10 | 2/10 |
| Task success rate | 10.0% | 20.0% |
| Tool selection proxy | 31.2% | 35.7% |
| Argument/schema validity | 94.1% | 94.7% |
| Invalid call rate | 5.9% | 5.3% |
| Average tool steps | 1.30 | 1.10 |
| Average model turns | 2.60 | 2.80 |
| Average tokens (input + output) | 4112.8 | 4471.8 |
| Average agent latency (seconds) | 6.93 | 10.64 |

Task-success difference: **+10.0 percentage points**.
Contract success is a narrow deterministic benchmark measure and still requires factual trace review. Tool-selection accuracy is the existing required-tool proxy, not an optimal-action oracle. Tokens include repeated input prompts; agent latency excludes initial model loading and the separate validator.

## Failure categories

```json
{
  "base": {
    "invalid_tool_call": 1,
    "task_contract_failure": 3,
    "validation_failure": 5
  },
  "adapter": {
    "invalid_tool_call": 1,
    "task_contract_failure": 3,
    "validation_failure": 4
  }
}
```

## Training and provenance

The selected training run used 15 approved continuation targets from 4 demonstrations for 300 updates/20 epochs.
The objective combines 75% first tool-name decision loss and 25% weighted completion loss, emphasizing changed code and JSON closure. Trainable LoRA weights use FP32; the frozen base uses NF4 with FP16 compute. Actual label, all-batch gradient, optimizer-update and reload audits are saved.
Rows SHA-256: `03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b`. Failed prefixes remain context only; the two truthful-failure demonstrations remain permanently excluded.
The benchmark contains 50 tasks split 40 support/10 held-out. No held-out targets or rewards were generated. Older held-out runs exist; this is a fixed split, not a claim of newly unseen tasks. Final task file bytes match the committed benchmark.

## Compute

Training: 1038.677 seconds on `Tesla T4`; peak CUDA allocation 8231636480 bytes. This is the selected run, not cumulative project spend.
Final evaluation hardware and runtime versions:
```json
{
  "python": "3.12.16",
  "platform": "Linux-6.12.115-140.210.amzn2023.x86_64-x86_64-with-glibc2.41",
  "machine": "x86_64",
  "cpu_count": 8,
  "torch": "2.11.0+cu130",
  "cuda_runtime": "13.0",
  "cuda_available": true,
  "gpus": [
    {
      "index": 0,
      "name": "Tesla T4",
      "memory_bytes": 15828320256,
      "compute_capability": "7.5"
    }
  ],
  "nvidia_driver_versions": [
    "580.178.04"
  ],
  "packages": {
    "transformers": "5.22.1",
    "accelerate": "1.15.0",
    "agenttune": "1.1.3",
    "huggingface_hub": "1.33.0",
    "tokenizers": "0.23.3"
  }
}
```
Actual credit balance and cumulative billing are not inferred from runtime. No closed-model cost/quality comparison was performed.

## AuditKIT and limitations

AuditKIT runs locally over real imported AgentTune EventLogs, without model inference, external judges or new tool execution. The outcome oracle independently recomputes the frozen contract from recorded validator and observation evidence. Its tool-validity and loop scores cover emitted actions, including finish; rejected outputs and blocked attempts are reported by the separate harness metrics.
These ten tasks and four demonstrations are too small to support a broad capability claim. Tasks share a small synthetic miniapp; the split is by task ID, not by unrelated repository. The benchmark uses keyword checks and narrow executable validators; passing may miss unrelated regressions. Correct END_RESPONSE stopping is a runtime efficiency correction, not evidence of SFT improvement. No DPO/GRPO or production safety guarantee is claimed.

## Evidence and reproduction

- [Complete paired records](../research/results/tiny-aya-final-heldout/20261115T071710Z.jsonl)
- [Frozen metadata](../research/results/tiny-aya-final-heldout/20261115T071710Z.metadata.json)
- [Independent audit](../research/results/tiny-aya-final-heldout/20261115T071710Z.audit.json)
- [AuditKIT report](../research/results/auditkit-final/20261115T071710Z.md)
- [Recorded walkthrough](demo.html)
- [Reproduction instructions](REPRODUCING.md)
- [Final protocol](../research/experiments/EXP-025-final-matched-heldout.md)

## Later support-only follow-up

After the matched held-out run, EXP-027 trained a distinct task-start adapter
from approved support targets and compared it on four known support tasks. The
base passed 0/4 contracts; the new adapter passed 2/4 (B021 and B041). For both
successful edits, the recorded full-file write matched the reviewed reference,
preserved unchanged AST nodes, and passed the named test plus a regression
test. B026 failed because the model looked for the test file at the wrong path;
B046 emitted an invalid finish action. These results are support-only and do
not establish held-out generalization or change the EXP-025 metrics/candidate.
See [EXP-027](../research/experiments/EXP-027-task-start-curriculum-plan.md)
and its saved training/replay audits.
