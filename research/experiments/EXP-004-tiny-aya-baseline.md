# EXP-004 — Untouched Tiny Aya held-out baseline

**Status:** Baseline and same-config trace-recovery run completed.
**Run ID:** `20261104T220436Z`
**Hugging Face Job:** `6aebab800e033f9c63203f50`
**Run kind:** Untouched model; no fine-tuning.

## Hypothesis

The untouched Tiny Aya Global model can use the workspace-scoped tools to
complete repository tasks on the fixed held-out benchmark. This run measures
the before-training performance; it does not test an adapter or changed prompt.

## Configuration and procedure

- Model: `CohereLabs/tiny-aya-global` at revision
  `af89d219b53ed9b13b8a4645f9c8028973510324`, FP16.
- Runtime: Python 3.12.16, PyTorch 2.11.0, Transformers 5.22.1,
  Accelerate 1.15.0, AgentTune 1.1.3.
- Hardware: Hugging Face Tesla T4, 15 GiB VRAM; CUDA 13.0 runtime.
- Frozen config: `terminal-slm-tiny-aya-eval-v1`, SHA-256
  `8a12b2f0774f5afb73f73fb4d12eb510cbc3fc952240715f6de102cb932138a1`.
- Prompt SHA-256:
  `44d65679d16e73b5fa4e5dae8c6d77bee81b1ead3ecdb6703498b37536d17cc5`.
- Tool schema SHA-256:
  `cb24a19bf8fea75e2e716ddc69cb9a2205916c740f29ef8266c931e2db679eb6`.
- Benchmark: `terminal-slm-initial-v1`, held-out split, IDs B005, B010, B015,
  B020, B025, B030, B035, B040, B045, and B050. No benchmark changes were
  made.
- Each task used a fresh workspace copy and the frozen validator. Decoding was
  greedy; limits were eight agent steps and 120 seconds per task.

## Results

| Metric | Result |
|---|---:|
| Tasks | 10 |
| Successful tasks | 0 / 10 |
| Task success rate | 0.0% |
| Tool selection accuracy | 15.8% |
| Argument/schema validity | 85.7% |
| Invalid tool-call rate | 14.3% |
| Average tool steps/task | 2.9 |
| Average model turns/task | 3.8 |
| Average tokens/task | 4,192.7 |
| Average latency/task | 11.09 s |

Failure categories: 3 `agent_incomplete`, 5 `invalid_tool_call`, 1
`task_contract_failure`, and 1 `tool_execution`.

| Task | Category | Outcome | Failure |
|---|---|---:|---|
| B005 | repository_navigation | Fail | agent_incomplete |
| B010 | function_discovery | Fail | agent_incomplete |
| B015 | configuration_discovery | Fail | task_contract_failure |
| B020 | environment_variable_discovery | Fail | agent_incomplete |
| B025 | simple_bug_fix | Fail | invalid_tool_call |
| B030 | failing_test_fix | Fail | invalid_tool_call |
| B035 | validation_change | Fail | invalid_tool_call |
| B040 | api_route_fix | Fail | invalid_tool_call |
| B045 | configuration_update | Fail | invalid_tool_call |
| B050 | explaining_test_failure | Fail | tool_execution |

## Failure analysis

The complete recovery traces show repeated navigation errors: B005 repeatedly
read `.` as though it were a file, alternating with `ls` without progressing;
B010 and B020 repeatedly called `read_file` on a directory or nonexistent
path until reaching the step limit. B030 searched and read the target, then
emitted a `shell` action with an unsupported `args` field while attempting an
`rm` command; argument validation rejected it. B025, B035, B040, and B045
failed to emit a parseable AgentTune Cohere-fallback action (B025 and B035 did
so after an observation; B040 and B045 failed on the initial response). B015
used a shell search and returned an answer that did not satisfy the task
contract. B050 ran the test suite, which failed, but the final answer
mischaracterized the failure.

The model can load and run on the T4, and the selected actions are mostly
schema-valid, but tool relevance, recovery, and executable task completion are
poor. At this baseline, valid syntax does not translate to reliable task
execution. Do not train against held-out tasks; use only support tasks to
develop and validate any future rollout or reward logic.

## Artifact integrity and trace-recovery run

The job's aggregate summary and environment record were intact. Hugging Face
truncated two long task-result lines at its console line limit, so the original
run does not have a complete raw trajectory archive. The extractor preserved
the eight complete records as a **partial** JSONL and saved explicit status:

- Aggregate summary: `research/results/tiny-aya-baseline/20261104T220436Z.summary.json`
- Runtime metadata: `research/results/tiny-aya-baseline/20261104T220436Z.metadata.json`
- Partial traces (8/10): `research/results/tiny-aya-baseline/20261104T220436Z.partial.jsonl`
- Missing-trace status: `research/results/tiny-aya-baseline/20261104T220436Z.artifact-status.json`
- Original job log: `research/results/tiny-aya-baseline/20261104T220234Z.log`

The evaluator was changed to emit bounded base64 chunks, then rerun with the
same frozen model, prompt, tasks, schemas, scoring, limits, and package versions.
This is a separate trace-recovery run, not a replacement or merge with the
first run:

- Recovery run ID: `20261104T221629Z`
- Hugging Face Job: `6aebae3c0e0338d1bb8040ad`
- Complete 10-task traces:
  `research/results/tiny-aya-baseline/20261104T221629Z.jsonl`
- Recovery metadata and summary:
  `research/results/tiny-aya-baseline/20261104T221629Z.metadata.json` and
  `research/results/tiny-aya-baseline/20261104T221629Z.summary.json`
- Recovery job log:
  `research/results/tiny-aya-baseline/20261104T221412Z.log`

The recovery run reproduced the same success rate, selection accuracy,
argument/schema validity, invalid-call rate, steps, turns, tokens, and failure
counts. Mean latency was 11.123 seconds/task versus 11.092 seconds/task in the
first run. All ten ordered held-out IDs are present in the recovery JSONL, and
each record contains both the Terminal-SLM trajectory and complete AgentTune
trace. The complete recovery trajectories are suitable for failure analysis;
they are clearly identified as a second run rather than misattributed to the
first run.

## Conclusion and next experiment

The untouched-model baseline is 0/10 on the fixed held-out evaluation, and an
identical recovery run reproduced the aggregate result with complete traces.
The next phase is support-only rollout and reward validation. Use the held-out
traces for reporting and diagnosis only; do not generate training data, tune
prompts, or select rewards from held-out examples. Do not alter the benchmark.
