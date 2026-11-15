# Terminal-SLM executable benchmark v1

This is the initial Phase B task set. It contains 50 deterministic repository
tasks over the pinned `snapshots/miniapp` Python project. The task files define
inputs and checks only; no model run has been made and no result data is
included.

## Categories

There are five tasks in each category:

- Repository navigation
- Function discovery
- Configuration discovery
- Environment-variable discovery
- Simple bug fixes
- Failing test fixes
- Validation changes
- API route fixes
- Configuration updates
- Explaining test failures

## Task format

Tasks are JSON Lines records in `train_support.jsonl` and `heldout_eval.jsonl`.
Each record contains:

| Field | Meaning |
| --- | --- |
| `task_id` | Permanent ID (`B001`–`B050`). |
| `category` | One of the categories above. |
| `split` | `train-support` or `held-out-eval`. |
| `snapshot` | Path to the immutable repository snapshot. |
| `snapshot_sha256` | Digest over sorted relative paths and file bytes. |
| `instruction` | User request passed to the agent. |
| `expected_outcome` | Human-readable description of the expected behavior/change. |
| `validation_command` | Argument vector run in the isolated task workspace. |
| `success_condition` | Required final-answer terms, tool names, accepted validation exit codes, optional tool-output evidence, and file-change constraints. |
| `reference_patch` | Optional patch text; omitted in this initial set. |

`load_benchmark(split)` validates IDs, split assignment, snapshot containment,
and the snapshot digest before returning tasks. Use `training_task_ids()` when
preparing any future training examples; it returns IDs from `train-support`
only. `run_task(task, agent)` copies the snapshot into a new workspace, invokes
the agent there, and runs the task's validation command there. Callers should
place workspaces under a temporary parent and remove them after retaining any
approved trajectory artifact.

## Frozen split and contamination prevention

The split was fixed before any SFT/DPO/GRPO data preparation. Each of the ten
categories contributes four train-support tasks and one held-out task, for 40
train-support and 10 held-out tasks (80/20). The permanent held-out IDs are
listed in `manifest.json`; do not reassign or reuse them as prompts, examples,
preference pairs, reward demonstrations, or training-derived debugging data.
Only train-support task IDs may be used for training-data generation. Keep
held-out trajectories and outcomes in the evaluation store and do not feed them
back into training. Changes to this set require a new benchmark version and a
new held-out set.

The loader returns only the requested split, and the dedicated
`training_task_ids()` helper cannot return held-out IDs. Repository access is
not an access-control boundary: project maintainers must also follow the
separation policy and keep held-out artifacts out of training pipelines.

## Running and interpreting

The framework is model-agnostic. A caller supplies an agent function that
receives the task instruction and isolated workspace and returns either final
text or an `AgentResult`. The runner does not start inference on import or when
loading tasks. Validation uses a clean, workspace-scoped environment and a
timeout. The evaluator aggregates only records produced by actual runs; its
unit tests use isolated fixtures and do not persist evaluation results.

The standalone `evals.benchmark.evaluator` is an earlier local Transformers
integration retained for framework development; it is not the runner used to
produce the current hackathon results. The reported cloud baseline and paired
evaluation use the versioned AgentTune path in `training/agenttune/`, described
in [the reproduction guide](../../submission/REPRODUCING.md). The current
results and actual task trajectories are linked from
[submission/RESULTS.md](../../submission/RESULTS.md).

Task IDs are checked against the selected split before model initialization,
so a held-out ID cannot be selected from a support run. Results go to
split-specific directories under `evals/benchmark/results/` unless explicit
paths are supplied. Each JSONL row
includes the complete agent trajectory, validation output, outcome, failure
category, and per-task metrics. Records include model/config fingerprints,
package versions, quantization/dtype, prompt text and hash, tool schemas and
hash, benchmark version and held-out IDs, hardware/CUDA/driver details, and run
limits. A Markdown summary and a JSON summary are written alongside the JSONL.
When `--device cuda` is selected, evaluation refuses to proceed if PyTorch
cannot see a CUDA device. The temporary workspace is removed after the JSONL
row is written.

Metric definitions:

- **Tool selection accuracy:** required tool selections found divided by the
  required selections plus distinct unexpected tools selected.
- **Argument/schema validity:** valid parsed tool-call blocks divided by
  emitted `<tool_call>` blocks. Rejected blocks count as invalid.
- **Invalid tool-call rate:** rejected blocks divided by emitted blocks.
- **Steps:** successfully executed tool calls; model turns are also reported.
- **Tokens:** exact prompt plus generated token counts accumulated by
  `LocalTransformersClient` over every turn. Other clients with no token
  counters are reported as unavailable, not estimated.
- **Latency:** agent plus task validation wall time, excluding snapshot copy.
- **Failure categories:** agent runtime/inference, malformed tool call,
  timeout, tool execution, validation, incomplete agent, or task-contract
  failure.

Success conditions may also require `validation_output_contains` terms; these
are matched against combined stdout/stderr. This is used by test-explanation
tasks so an unrelated nonzero exit cannot count as the expected failure.
They may also declare `tool_evidence` entries containing a tool name,
`command_contains` terms, `arguments_match` exact argument values, and
`result_contains` terms. All specified conditions must match the same executed
tool call. B016 and B046 require reading the expected source file and observing
the function body; B046 separately requires the agent's own test run to observe
`AssertionError: 10 != 25`. The benchmark runner's separate validation output
cannot satisfy that evidence. B046 also fails if any task file changes.

On 2026-11-10, selected support pilot tasks were clarified after rollout
evidence: B001 asks grep to find the `ROUTES` definition rather than search file
names; B006 requires `read_file` in addition to `grep`; B011 and B016 name the
exact settings/source file and require reading the implementation; B021, B026,
B031, B036, and B041 identify the target file and acceptance test; and B046
requires the tool evidence above. These revisions apply only to `train-support`;
held-out records and the frozen baseline config were not changed. See EXP-005.

The initial miniapp is a compact framework smoke corpus, not a claim of broad
coverage across real repositories. Review task difficulty and repository
diversity before freezing a larger benchmark version.
