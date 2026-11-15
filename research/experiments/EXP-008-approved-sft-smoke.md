# EXP-008 — Approved-reference SFT pipeline smoke

Status: V2 GPU pipeline smoke completed on 2026-11-13. Behavioral improvement unproven.

## Hypothesis

Tiny Aya can receive finite QLoRA updates from the four owner-approved
references while preserving the Cohere JSON action protocol. This checks
training, adapter serialization, and reload. It does not test generalization.

## Data and configuration

Only G-B021-v1, G-B026-v1, G-B041-v1, and G-B046-v1 are admitted. Conversion
checks the owner's review hash against the original row and rejects mutations
or approval mismatches. Four trajectories yield 16 next-action examples.
Rejected v1 references and pending v2 references are excluded. Observations and
prompts are context only; loss applies to the single assistant completion.

Model revision, runtime and optimizer settings are in
`configs/tiny_aya_sft_smoke.json`. The smoke uses eight optimizer steps, batch
size one, seed 42, NF4 with double quantization, float16 compute, and rank-8
LoRA on q_proj/v_proj. Contexts over 4096 tokens cause failure rather than
truncation. A fresh private artifact repository is required before launch.

AgentTune 1.1.3 source was inspected at commit
`d4db4c12d52342227c4113a70183d0f64aa2f345`. `Project.train(fmt='sft')` delegates
to a TRL trainer factory. Reviewed JSON actions are TEXT events, because
TOOL_CALL events would serialize them as `<tool_call>` tags. The factory
asserts that AgentTune preserves the reviewed conversation, restores its system
message, and creates the same 16 prompt/completion rows as local conversion.
No policy log probabilities are invented for these authored demonstrations.

Reference implementation:
https://github.com/Lexsi-Labs/AgentTune/blob/d4db4c12d52342227c4113a70183d0f64aa2f345/examples/agentic_distillation_real.py

## Procedure

Build a bundle containing only approved references, matching original rows and
reviews, tool schemas, smoke config, and training code. It contains no benchmark
tasks or repository snapshots. Launch on HF t4-small with a 25-minute hard
limit. Require CUDA, completion masks, finite training loss, changed LoRA B
weights, saved adapter reload, and one short support prompt generation. Save
actual package versions, metrics, quantization, source IDs, config and row
hashes with the adapter in the private destination. No held-out evaluation is
part of this job.

## Results

### First cloud attempt and precision fix

Job `6af7852d23985fb9cd808642` prepared all 16 rows, then failed during the
first optimizer step with BF16 gradient unscaling under FP16 AMP. No completed
optimizer step or adapter improvement is claimed. Tokenizer EOS alignment was
an informational warning, not the exception.

Inspection of TRL v1.7.1 `sft_trainer.py` lines 1030–1037 shows it casts all
trainable parameters of quantized models to BF16 regardless of the configured
FP16 mode. V2 normalizes only trainable LoRA parameters to FP32 after trainer
initialization and before optimizer creation, preserving the frozen NF4 base
and FP16 compute. It rejects unexpected trainables or preexisting gradients
and logs the dtype counts. The retry below verified the correction on CUDA.

The failed job created an empty private repository. V2 permits reuse of that
empty private destination, but rejects public or populated destinations.

Source: https://github.com/huggingface/trl/blob/v1.7.1/trl/trainer/sft_trainer.py#L1030

Local conversion produced four trajectories and 16 action targets. Approval,
mutation, target masking structure, and bundle isolation tests passed.
The complete local suite passed: 79 tests; Python compilation, shell syntax,
and diff checks passed.
### Successful V2 pipeline run

Job `6af787e829dd05ff0365a16a`, run `20261113T220511Z`, completed on Tesla T4.
The result was extracted from the local full job log, preserving the payload
in `research/results/tiny-aya-sft-smoke/20261113T220511Z.result.json`. The adjacent
provenance file records the source log hash and the probe parsing failure.
The source example IDs and row hash match the local approved export.

- Eight training steps, four reference trajectories, 16 action targets, epoch 0.5.
- Training runtime: 25.7193 seconds; aggregate train loss: 0.894569767639041.
- LoRA weights changed and the saved adapter reloaded successfully.
- 1,916,928 trainable elements were normalized from BF16 to FP32.
- Peak allocated CUDA memory: 6,232,897,024 bytes (about 5.8 GiB).
- NF4 with double quantization and FP16 compute; no held-out tasks accessed.
- Runtime: torch 2.11.0, transformers 5.22.1, agenttune 1.1.3, trl 1.7.1,
  peft 0.21.3, datasets 5.2.0, bitsandbytes 0.50.2, accelerate 1.15.0.
- Private artifacts: https://huggingface.co/devjuneja789/terminal-slm-sft-smoke-v1/tree/main/20261113T220511Z

The sole post-training probe emitted a proposed shell `sed` command rather
than the demonstrated initial read. Its JSON string contains invalid `\\(`
escapes: strict parsing fails with `Invalid \\escape` at character 67. The
probe was only generated, not executed. It establishes neither a successful
tool call nor a completed task. A finite aggregate loss and changed adapter
weights are pipeline evidence; they do not demonstrate behavioral improvement
or a falling loss trend.

## Next step

Run a bounded paired support-only replay of the four approved tasks using the
base and this adapter with identical NF4 inference, prompt, schemas, decoding,
and limits. Capture complete tool trajectories and validators. This measures
reproduction of training examples, not generalization. Do not compare its NF4
score directly against the previous FP16 baseline. Use the replay to diagnose
invalid JSON, failure to edit, and completion behavior before longer training.
Keep the frozen held-out evaluation for a later defined model comparison.
