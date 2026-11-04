# EXP-002: Tiny Aya Global Phase 1 smoke

## Hypothesis

Tiny Aya Global can load and generate on the Hugging Face GPU Job, and can
complete one workspace-scoped repository discovery task through AgentTune.

## Configuration

- Model: `CohereLabs/tiny-aya-global`
- Task: B006 (`train-support` only)
- AgentTune spine: `Project`, `ReActStrategy`, and `DictToolHarness`
- Tool interface: AgentTune's Cohere JSON-list fallback, adapted to the
  existing workspace-scoped `Toolbox`
- Job name: `terminal-slm-phase1-tiny-aya-smoke`
- Job ID: `6ae8ef4cc78863f9c630f409`
- Inference completed in 59.486 seconds according to the returned record.

The returned record did not include the installed package versions or GPU
model, so those details are unknown for this run.

## Attempts

### Attempt 1 — failed

**Failed.** Plain-text generation worked and returned a coherent sentence about
grep. The tool episode returned three outputs: first a `git` command attempting
to run grep against the nonexistent `src` path, then an irrelevant `git
--version`, then malformed JSON that embedded a fragment of the `read_file`
schema as an argument value. The policy parser rejected the third output.

No final answer or successful required `grep` action was recorded. The task's
separate package-import validation command exited 0, but it does not validate
the requested discovery; that pass does not count as task success. AgentTune
raised before returning its EventLog, so its trajectory is empty. The raw model
outputs supplied by the user are preserved in
`research/results/phase1_tiny_aya_smoke_6ae8ef4cc78863f9c630f409.jsonl`.

## Failure diagnosis

- **Model inference:** functioning; plain-text generation succeeded.
- **Tool-call formatting/schema following:** unreliable. The model emitted a
  syntactically valid but semantically wrong first action, an irrelevant
  second action, and then malformed JSON.
- **Prompt:** likely contributed. The previous prompt serialized full JSON
  schemas, which may have encouraged the model to copy schema descriptions
  into argument values. This is an inference, not proven by one run.
- **Parser:** correctly rejected the malformed third action; it did not execute
  it.
- **Executor:** not shown to have failed; the returned record contains no
  trajectory or execution details to establish which calls executed.
- **Context formatting:** not yet isolated. The previous smoke did not save the
  rendered messages, so this run cannot distinguish model weakness from
  context-formatting effects.
- **Benchmark validator:** B006's validation command checks package imports,
  not the discovery answer. The smoke also checked required tool selection and
  response terms, so it correctly remained a failure.

## Harness changes before retry

The smoke prompt now presents compact tool signatures instead of full nested
schemas, clarifies when to use `grep`, and includes one minimal correctly
shaped action example. The runner also records rendered prompts, parse errors,
and tool executions so the next failure can be diagnosed. No benchmark task,
held-out task, model weight, or training data was changed.

### Attempt 2 — tool loop passed; task response incomplete

- Job ID: `6ae8f1b6c78863f9c630f5e6`
- Task: B006 (`train-support`)
- Elapsed time: 53.959 seconds
- Plain-text generation: non-empty and coherent.
- Tool action: valid `grep(pattern="def resolve_route", path=".")`.
- Tool result: successful match at `shop/api.py:13`.
- Final response: named the correct file and described the function signature.
- Parser: both actions parsed; no parse errors.
- The runner marked its configured smoke predicate passed; validation command
  exited 0.
- Raw trajectory: `research/results/phase1_tiny_aya_smoke_6ae8f1b6c78863f9c630f5e6.jsonl`.

On review against the full user instruction, the answer gave the location and
signature but did not explain the function's purpose. The smoke's initial
success predicate checked only for the function name, path, grep selection,
successful observation, and validator; it did not check semantic completeness.
Therefore this run proves that the model loads and the AgentTune tool loop
works, but it does not yet meet the full task-completion gate. The smoke now
requires a purpose statement describing how method and path map to a route, and
prompts the agent to inspect implementation when a signature is insufficient.

The changed prompt and successful tool selection are consistent with the
previous schema-heavy prompt contributing to failure, but two attempts do not
establish that as the sole cause. GPU model and installed package versions
were not included in the returned result, so they remain unrecorded.

### Attempt 3 — failed semantic check

- Job ID: `6ae8f3110e0338d1bb7e9222`
- Task: B006 (`train-support`)
- Elapsed time: 53.697 seconds
- Plain-text generation, `grep`, observation, and schema parsing all worked.
- The model returned the same signature-only answer and did not call
  `read_file`; the strengthened purpose predicate correctly failed.
- Validation exited 0 but only checks package importability.
- Raw trajectory: `research/results/phase1_tiny_aya_smoke_6ae8f3110e0338d1bb7e9222.jsonl`.

The prompt explicitly stated that a signature is insufficient and instructed
the model to inspect matching source lines with `read_file`; it still finished
after `grep`. This points to instruction-following behavior in the current
Tiny Aya/prompt setup. It is not evidence of parser, tool schema, or executor
failure. Since the episode runner has no completion gate, the model is free to
finish prematurely; the smoke correctly detects that post hoc.

## Next step

### Attempt 4 — failed tool-argument recovery

- Job ID: `6ae8f5350e0338d1bb7e93b8`
- Task: B006 (`train-support`)
- Elapsed time: 72.554 seconds
- `grep` succeeded; five subsequent `read_file` calls failed because the model
  repeatedly supplied `end_line: 0` (and once `start_line: 0`).
- The policy stopped at six steps without a final answer. Validation exited 0
  but checks package importability only.
- Raw selected trajectory and all six raw completions:
  `research/results/phase1_tiny_aya_smoke_6ae8f5350e0338d1bb7e93b8.jsonl`.

The completions contained multiple JSON action arrays each. The policy selected
the first array for execution but then replayed the entire raw completion in
the next model prompt. That exposed discarded proposed actions alongside the
actual tool observation and likely contributed to the confused repeated calls.
The compact prompt also omitted the schema's minimum line-number constraints,
even though the tool error clearly reported the invalid value. The harness now
retains raw completions for audit but feeds only the selected action back as an
assistant turn, and it includes numeric bounds/defaults in the compact tool
signatures.

### Attempt 5 — B002 answer discarded by adapter

- Job ID: `6ae8f6fec78863f9c630f9f0`
- Task: B002 (`train-support`)
- Elapsed time: 62.111 seconds
- `grep` found the route-table documentation at `shop/api.py:1`.
- On the next generation, Tiny Aya proposed `read_file` and also emitted the
  correct plain-text answer: “The route table is defined in `shop/api.py` at
  line 1.”
- The then-current adapter accepted only the JSON action, discarded the
  adjacent plain-text answer, and continued making five redundant reads until
  the step limit. No `finish` event was produced, so the run was correctly
  marked failed by that adapter.
- Raw selected trajectory and model outputs:
  `research/results/phase1_tiny_aya_smoke_6ae8f6fec78863f9c630f9f0.jsonl`.

This is a context/protocol adaptation issue rather than a failure to identify
the answer. The adapter now converts a plain-text tail to AgentTune's `finish`
action only after at least one tool observation; it records the adaptation and
retains the full raw completion. A fresh cloud run is still required to verify
the change.

### Attempt 6 — passed basic cloud smoke

- Job ID: `6ae8fbdac78863f9c630fde2`
- Task: B002 (`train-support`)
- Elapsed time: 51.741 seconds
- Plain-text generation succeeded.
- `grep(pattern="route table", path=".")` returned the declaration comment
  at `shop/api.py:1`.
- Tiny Aya then emitted a redundant `read_file` proposal followed by the
  correct plain-text answer. The updated adapter converted the text to an
  AgentTune `finish` action and discarded the unexecuted proposal.
- The final response satisfied B002's path condition; the required tool,
  successful observation, and validation command also passed.
- Raw trajectory: `research/results/phase1_tiny_aya_smoke_6ae8fbdac78863f9c630fde2.jsonl`.

This meets Phase 1's basic cloud smoke gate. It establishes that Tiny Aya can
load, make a valid tool call, receive the observation, and complete a simple
repository task through the AgentTune adapter. It does not erase the B006
failures or demonstrate robust function-purpose discovery. Hardware and
installed package versions were not included in the result; capture them in
the formal baseline.

## Next phase

Proceed to Phase 2: freeze the Tiny Aya prompts, tool descriptions, generation
settings, limits, task validators, and split boundaries; confirm the held-out
IDs remain excluded from all training inputs. Then run the untouched Tiny Aya
held-out baseline (Phase 3) in the cloud. Do not start training before the
baseline is recorded.
