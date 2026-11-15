# EXP-027: Complete-trajectory task-start curriculum

## Question

Would adding reviewed task-start actions to the admitted continuation targets
improve completion from a fresh support-task context?

This is a support-only coverage diagnostic. It is not a held-out evaluation,
not proof of generalization, and not a claim that longer training improves
reliable repository repair.

## Fixed protocol and provenance

The curriculum combines 16 actions from four previously approved complete
support demonstrations with 15 separately admitted corrective targets: 31
distinct reviewed action targets total. The two truthful-failure demonstrations
remain audit-only. Failed model prefixes are context, never labels. No held-out
task, patch, reward, or trajectory was included in training.

Before optimization, the trainer checked all 31 tokenized targets, label masks,
source indices, JSON/closing-token labels and finite gradients. A first job
(`6af99b6123985fb9cd83452a`) stopped before optimizer creation because a legacy
assertion expected three write targets while the mixed set has six. The
assertion was corrected for the mixed set while preserving the legacy check;
focused regression tests cover both cases. This is an implementation failure,
not a training result.

The fixed retry used the same data, config, seed and budget: 620 optimizer
updates / 20 epochs on a Hugging Face Tesla T4 job. Its 31-batch gradient
preflight passed at AMP scale 256 with finite gradients, no preflight updates,
and cleared gradients. The adapter changed and reloaded successfully.

- Training run: `20261115T115312Z`
- HF job: `6af99cb423985fb9cd83467e`
- Adapter artifact: `devjuneja789/terminal-slm-tiny-aya-task-start-620-v1`
- Adapter artifact commit: `c1e8bbbcf08a32985a0471633c699f5a68b64cfe`
- Training result: [`20261115T115312Z.result.json`](../results/tiny-aya-task-start-sft/20261115T115312Z.result.json)
- Training loss: `0.1771143303586277`; this is not a task-success metric.

## Paired support replay

The base and new adapter were run on the same four known support tasks with the
same frozen inference/runtime configuration. The exact JSONL trajectories,
metadata, summary and local audit are saved under
[`research/results/tiny-aya-task-start-replay`](../results/tiny-aya-task-start-replay).

| Model | Contract passes | Tool-selection accuracy | Schema validity | Mean latency |
|---|---:|---:|---:|---:|
| Base | 0/4 | 50.0% | 83.3% | 14.38 s |
| Task-start adapter | 2/4 | 63.6% | 83.3% | 20.58 s |

The adapter passed B021 and B041. Both successful `write_file` actions matched
the reviewed full-file references exactly, preserved all unchanged Python AST
nodes, changed only `shop/config.py`, and passed their named tests plus the
page-size upper-bound regression test. B026 failed before writing because the
model searched for the test file at the wrong path. B046 emitted an invalid
finish action and did not complete its requested explanation.

## Interpretation and decision

Adding start-of-task coverage improved this four-task support replay from 0/4
to 2/4, with two independently checked edits. The other two tasks still failed.
This small known-task replay does not establish held-out generalization. The
new adapter was not evaluated on held-out tasks and does not replace or alter
the adapter and matched held-out results in EXP-025. The outcome supports a
narrow claim about learning two reviewed support tasks; it does not justify a
broad claim of reliable terminal coding. No further training or inference is
recommended for this submission.
