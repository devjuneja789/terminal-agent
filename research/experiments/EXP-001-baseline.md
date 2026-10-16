# EXP-001: Untuned Qwen3-1.7B held-out baseline

## Hypothesis

The untouched Qwen3-1.7B model, connected to the Terminal-SLM harness through
Qwen3's native tool-call format, can complete a useful fraction of the fixed
held-out repository tasks with valid tool calls and successful task validation.

## Configuration

- Model: local `./models/Qwen3-1.7B`, `Qwen3ForCausalLM`, model type `qwen3`.
- Checkpoint config SHA-256: `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197`.
- Model revision/commit: not available in the local checkpoint metadata. Weights
  were not modified or fine-tuned. The checkpoint files and sizes are recorded
  in the raw run metadata; weight-file content hashes were not recorded.
- Inference: CUDA, float16, no quantization, greedy decoding (`do_sample=false`),
  512 maximum generated tokens, thinking disabled, local files only, KV cache on.
- Software: PyTorch `2.14.0+cu126`, Transformers `5.19.0`, Tokenizers `0.23.3`.
- Hardware: WSL2 Linux, NVIDIA GeForce GTX 1060 Max-Q (6,442,319,872 bytes
  reported VRAM, compute capability 6.1), driver `582.78`, CUDA runtime `12.6`,
  12 CPU cores, 8,217,903,104 bytes RAM.
- Prompt: `DEFAULT_SYSTEM_PROMPT`, SHA-256
  `a715a2ea1a78b7ba910f1dcd2e1fea740a1ace525d262a328eeba1374042075a`.
- Tool schemas: the five standard OpenAI-style JSON schemas for `shell`,
  `read_file`, `write_file`, `grep`, and `git`; SHA-256
  `cb24a19bf8fea75e2e716ddc69cb9a2205916c740f29ef8266c931e2db679eb6`.
- Agent limits: 16 tool calls, 600 seconds maximum execution time per task, and
  120 seconds for validation.

## Dataset and benchmark version

- Benchmark: `terminal-slm-initial-v1`.
- Split: fixed `held-out-eval`, 10 tasks: `B005`, `B010`, `B015`, `B020`,
  `B025`, `B030`, `B035`, `B040`, `B045`, `B050`.
- Repository snapshot SHA-256:
  `64a99972f1e042cd9f1708e13a865fc1416977af190f3e7124a4ee484d0474a5`.
- No benchmark tasks or success conditions were changed for this run. No
  held-out data was used to create training data.

## Evaluation procedure

The benchmark evaluator created an isolated workspace for each task, initialized
the task's fixed repository snapshot, ran the agent, recorded its trajectory,
and ran the task validation command. Success required both the task's response
and required-tool conditions and its validation condition. The evaluation used
the same general prompt, schemas, and harness as the support runs; no prompt was
changed for individual held-out tasks. This was inference only: no training or
weight updates occurred.

The complete task records, trajectories, hardware and model metadata are in
[`run_20261016T195242Z.jsonl`](../../evals/benchmark/results/held-out-eval/run_20261016T195242Z.jsonl).
The evaluator's human-readable summary is
[`run_20261016T195242Z.md`](../../evals/benchmark/results/held-out-eval/run_20261016T195242Z.md),
with aggregate machine-readable metadata in
[`run_20261016T195242Z.json`](../../evals/benchmark/results/held-out-eval/run_20261016T195242Z.json).

## Results

| Metric | Result |
|---|---:|
| Tasks | 10 |
| Successful tasks | 1 |
| Task success rate | 10.0% |
| Tool selection accuracy | 17.6% |
| Argument/schema validity | 100.0% |
| Invalid tool-call rate | 0.0% |
| Average tool steps per task | 2.50 |
| Average model turns per task | 3.50 |
| Average tokens per task | 4,554.8 |
| Average latency per task | 72.781 seconds |

Failure categories reported by the evaluator:

- `tool_execution`: 7 tasks
- `validation_failure`: 2 tasks
- Passed: 1 task

Per-task outcomes:

| Task | Category | Result | Failure category | Latency (s) |
|---|---|---:|---|---:|
| B005 | Repository navigation | Fail | `tool_execution` | 277.896 |
| B010 | Function discovery | Pass | — | 17.446 |
| B015 | Configuration discovery | Fail | `tool_execution` | 313.298 |
| B020 | Environment-variable discovery | Fail | `tool_execution` | 11.506 |
| B025 | Simple bug fix | Fail | `validation_failure` | 16.853 |
| B030 | Failing-test fix | Fail | `tool_execution` | 12.110 |
| B035 | Validation change | Fail | `tool_execution` | 15.441 |
| B040 | API route fix | Fail | `tool_execution` | 12.158 |
| B045 | Configuration update | Fail | `validation_failure` | 35.908 |
| B050 | Explain test failure | Fail | `tool_execution` | 15.195 |

## Failure analysis

- **Tool schemas and parser:** All attempted calls passed schema validation;
  there were no malformed or invalid calls. This run provides no evidence of a
  parser or schema-validation failure.
- **Tool execution and workspace grounding:** Several tasks failed because the
  model supplied paths that did not exist in the task workspace (for example,
  `project`, `config`, and `configurations.env`). The tools returned explicit
  path errors. In these cases the executor enforced the workspace boundary and
  reported the error; the model did not recover by inspecting the actual
  workspace structure.
- **Loop behavior:** B015 repeatedly guessed paths and exhausted the 16-call
  limit. Other tasks often stopped after one unsuccessful tool call, despite
  receiving the error observation. This points to weak recovery and task
  completion behavior in the model/prompt interaction, rather than invalid
  tool syntax.
- **Required actions and validation:** On B025, the model described an intended
  code change but did not write it or run the required validation. On B045 it
  located relevant timeout references but did not perform the required update
  with `write_file`; validation consequently failed. B030, B035, and B040 also
  did not perform the required shell/write actions.
- **Latency:** B005 and B015 took 277.9 and 313.3 seconds. B015 used all 16 tool
  calls. This suggests substantial latency and repeated unsuccessful reasoning
  on some cases, even though the rest of the run was much faster.
- **Benchmark integrity:** The fixed snapshot's validation commands and
  evaluator completed, and the raw records show no evidence that a benchmark
  defect caused these failures. The benchmark has not been changed after seeing
  these results.

## Conclusion

The untouched model completed 1 of 10 held-out tasks (10.0%). Tool-call syntax
and argument schemas were reliable in this run, but tool selection, repository
grounding, error recovery, and execution of requested edits were not. The
baseline therefore does not support the hypothesis that the current base model
and harness can reliably complete the held-out task set. This result is a small
10-task benchmark estimate; it should be interpreted together with the fixed
task list and raw trajectories rather than as a broad estimate of coding-agent
performance.

## Next experiment

Before any training, review the trajectories and fix only confirmed harness or
benchmark defects. Do not tune against the held-out task wording or use these
tasks to generate training examples. If the benchmark and harness remain
unchanged, preserve this run as the baseline. The next model experiment should
use the train-support split for development and data generation, then evaluate
once on the permanently held-out split with this same prompt, schemas, limits,
and evaluator. Record any harness or configuration change as a new experiment
so results remain comparable.
