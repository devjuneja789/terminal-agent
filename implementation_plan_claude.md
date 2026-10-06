# Terminal-SLM: Implementation Plan

## 0. Project goal

Build a small language model specialized for reliable terminal/tool use, then embed it inside a local terminal agent and evaluate it on executable coding tasks.

The project has three connected layers:

```text
                 POST-TRAINING
Base Qwen3-1.7B
      |
      +--> SFT + QLoRA
      |
      +--> optional DPO
      |
      +--> optional GRPO
      |
      v
Tool-Calling SLM
      |
      v
                 AGENT HARNESS
CLI -> model -> tool call -> executor -> observation -> model -> ...
                              |
                              v
                 EVALUATION / RL ENVIRONMENT
                 tests + task success + rewards
```

The goal is NOT to train a general-purpose model. The goal is to make a small model reliably perform a constrained set of terminal actions.

---

## 1. Hardware-aware strategy

### Local machine

GPU:
- NVIDIA GeForce GTX 1060 Max-Q
- 6 GB dedicated VRAM
- Pascal generation / Compute Capability 6.1

### What this means

Do **not** plan around full-precision training on the laptop GPU.

Use this split instead:

| Work | Where |
|---|---|
| Dataset creation | Local |
| Agent harness development | Local |
| Evaluation harness | Local |
| Small inference tests | Local |
| SFT / QLoRA training | Cloud GPU |
| DPO | Cloud GPU |
| GRPO | Cloud GPU, preferably >=16 GB VRAM; 24 GB is more comfortable |
| Final quantized inference | Local GTX 1060 |
| Heavy/long-context fallback | Remote GPU endpoint |

Qwen3-1.7B's normal model files are about 4.08 GB in BF16. Unsloth publishes Qwen3-1.7B GGUF quantizations around 1 GB for Q4 variants, which is much more suitable for the GTX 1060. [Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B) / [Unsloth GGUFs](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)

The current bitsandbytes documentation lists NVIDIA Pascal (Compute Capability 6.0+) as supported for NF4/FP4 quantization, so the GTX 1060 is in the supported range for 4-bit quantization. Treat training on this GPU as a fallback experiment rather than the primary training setup because memory and kernel/software constraints make it much less attractive than a cloud GPU. [bitsandbytes hardware support](https://huggingface.co/docs/bitsandbytes/installation)

---

## 2. Model choice

### Initial model

Use:

```text
Qwen/Qwen3-1.7B
```

Prefer the instruction/chat-oriented checkpoint for the first tool-calling experiment rather than starting from a raw base model. Preserve its instruction-following ability and teach the additional terminal/tool protocol through SFT.

Start with the 1.7B model. Do not jump to 4B/8B until the training and evaluation pipeline works end-to-end.

### Why 1.7B

- Small enough to train with QLoRA on rented/free cloud GPUs.
- Small enough to quantize and run locally.
- Large enough to make the project non-trivial.
- Gives a clean before/after comparison between base and post-trained behavior.

---

## 3. Define the tool protocol first

Do this before collecting training data.

Start with only 5 tools:

```text
shell(command, timeout)
read_file(path, start_line, end_line)
write_file(path, content)
grep(pattern, path)
git(command)
```

Add a sixth tool later only if the benchmark needs it.

### Tool-call format

Do not invent a custom tool-call syntax. Qwen3 already supports tool calling natively using a Hermes-style `<tool_call>` template, and llama.cpp's autoparser detects this format automatically for any GGUF model trained with tool support, with no extra server configuration.

Target this native format from day one:

- Define the 5 tools using the standard OpenAI-style `tools=[...]` JSON schema (name, description, parameters).
- Let the base model emit calls as `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
- Keep this format consistent across the harness, the SFT data, and the exported GGUF so nothing has to be re-mapped at inference time.

Benefits:

- SFT only has to *sharpen* an existing capability (reliable use of 5 specific tools) rather than teach tool-calling from scratch, which likely shrinks the amount of training data needed.
- The exported GGUF works out of the box with llama.cpp's OpenAI-compatible server (`--jinja`) with no custom parser to write or maintain.
- Qwen3 also has a thinking on/off toggle that affects output length and latency — decide explicitly which mode the SFT data (and eval) target, since it changes your own "avg tokens per task" and "latency" metrics.

### Safety boundaries

The `shell` tool should NOT simply execute arbitrary model output blindly.

Implement:

- working-directory sandbox
- command timeout
- maximum stdout/stderr size
- environment-variable filtering
- blocked destructive commands for the initial benchmark
- explicit command logging
- optional human approval mode

The agent should initially operate inside a temporary repository created for each task.

---

## 4. Build the terminal harness before fine-tuning

Repository structure:

```text
terminal-slm/
├── agent/
│   ├── loop.py
│   ├── prompts.py
│   ├── parser.py
│   └── state.py
├── tools/
│   ├── shell.py
│   ├── filesystem.py
│   ├── grep.py
│   └── git.py
├── model/
│   ├── client.py
│   └── local.py
├── data/
│   ├── raw/
│   ├── sft/
│   └── preference/
├── evals/
│   ├── tasks/
│   ├── runner.py
│   ├── metrics.py
│   └── rewards.py
├── training/
│   ├── sft.py
│   ├── dpo.py
│   └── grpo.py
├── configs/
│   └── *.yaml
├── scripts/
│   ├── export.py
│   └── benchmark.py
├── cli.py
└── README.md
```

The first working milestone is:

```text
User task
  -> model
  -> structured tool call
  -> tool executes
  -> result returned to model
  -> model decides next action
  -> final answer
```

Do this with the **base Qwen3-1.7B** before training anything.

---

## 5. Create the benchmark before creating the training set

Make a small benchmark of deterministic repository tasks.

Initial target: 50-100 tasks.

Examples:

1. Find where a function is defined.
2. Find an environment variable used by the application.
3. Fix a one-line Python bug.
4. Fix a failing unit test.
5. Update a function to satisfy a test.
6. Rename a variable across a small repository.
7. Add a missing validation check.
8. Fix an incorrect API route.
9. Update a configuration file.
10. Find and explain why a test fails.

Each task should contain:

```text
repository snapshot
user instruction
expected outcome
validation command
optional reference patch
```

Example:

```json
{
  "id": "py_bug_017",
  "instruction": "Fix the failing authentication test.",
  "repo": "tasks/py_bug_017/repo",
  "test_command": "pytest -q",
  "success_condition": "all targeted tests pass"
}
```

This benchmark becomes both your evaluation suite and, later, your RL environment.

### Held-out evaluation split

Before generating any SFT (or later DPO/GRPO) data, split the benchmark tasks into two fixed groups and never move tasks between them:

```text
train-support set   (~80% of tasks, e.g. 40 of 50)
held-out eval set   (~20% of tasks, e.g. 10 of 50)
```

- Only tasks in the **train-support set** may be used to generate SFT trajectories, DPO preference pairs, or GRPO reward-training rollouts.
- Only tasks in the **held-out eval set** are used for baseline, SFT, DPO, and GRPO evaluation (Sections 7, 9, 12) — every row in every results table comes from this same fixed set.
- Decide the split once, before touching Section 6, and record the held-out task ids somewhere durable (e.g. `evals/tasks/holdout_ids.txt`) so the split can't drift or get reshuffled later.

This is what keeps the eventual Base vs SFT vs DPO vs GRPO comparison honest — no method gets credit for having seen its own exam questions during data generation.

---

## 6. Build the SFT dataset

Write every trajectory in the native Hermes `<tool_call>` format defined in Section 3 — do not build a parallel/custom format for training data and reconcile it with the model's native format later.

Create tool-use trajectories such as:

```text
user task
-> assistant tool call
-> tool result
-> assistant tool call
-> tool result
-> final answer
```

Sources can be:

- hand-written high-quality trajectories
- trajectories generated by a stronger model and manually filtered
- your own demonstrations
- successful trajectories collected from an existing agent

Do NOT blindly dump generated traces into training.

Filter for:

- correct tool
- valid arguments
- minimal unnecessary actions
- useful observations
- correct final result

Target first version: roughly 1k-5k high-quality tool-use examples rather than a huge noisy corpus.

---

## 7. Baseline evaluation

Run the untouched Qwen3-1.7B through the benchmark.

Measure at least:

```text
Task success rate
Tool selection accuracy
Argument/schema validity
Invalid tool-call rate
Average steps per task
Average tokens per task
Latency
Failure categories
```

Save these numbers before training.

The project's central question becomes:

> Does post-training make a small model substantially more reliable at terminal tool use?

---

## 8. SFT + QLoRA

Primary training stack:

```text
Transformers
TRL
PEFT
bitsandbytes
Unsloth
```

Run this on a cloud GPU.

### Initial configuration

Start conservative:

```text
4-bit QLoRA
LoRA rank: 16 or 32
LoRA alpha: 32 or 64
LoRA dropout: 0.0-0.05
per-device batch size: 1-2
gradient accumulation: increase effective batch size
sequence length: 1024 first, then 2048 if memory allows
gradient checkpointing: enabled
```

Exact hyperparameters should be tuned after the first run rather than assumed up front.

### Trainable layers

Initially target the attention and MLP linear layers used by the model's transformer blocks. Compare all-linear targeting against attention-only if compute permits.

### First success criterion

The fine-tuned model should:

- emit valid tool calls reliably
- choose the right tool more often than the base model
- use correct argument schemas
- complete simple repository tasks without excessive steps

---

## 9. Evaluation after SFT

Run exactly the same benchmark again.

Produce a table like:

```text
                         Base       SFT-LoRA
Task success             XX%         XX%
Valid tool calls         XX%         XX%
Tool selection           XX%         XX%
Avg steps                XX          XX
Avg tokens               XX          XX
```

Do not fabricate target numbers. The real results become part of the README.

---

## 10. Add DPO only if there is a real preference signal

Do not add DPO just because the project mentions RL.

Construct preference pairs from actual failures:

```text
chosen:
  grep("MONGO_URI", ".")

rejected:
  read_file("node_modules/....")
```

or:

```text
chosen:
  run pytest -> inspect failing file -> patch -> rerun pytest

rejected:
  make many unrelated edits before inspecting the failure
```

The preference dataset should encode desirable agent behavior:

- fewer unnecessary actions
- safer commands
- correct tool selection
- better argument construction
- correct stopping behavior

Then run DPO and compare against SFT.

---

## 11. Add GRPO only after the executable environment works

This is the most interesting stage, but also the stage that needs the most compute and engineering discipline.

For each benchmark task:

```text
Task
 ↓
SLM generates a trajectory
 ↓
Harness executes tools
 ↓
Tests / validators run
 ↓
Reward computed
```

Possible reward components:

```text
+5  task solved
+2  targeted tests pass
+1  correct tool selection
+1  valid arguments
+1  useful intermediate observation
-1  unnecessary tool call
-2  malformed tool call
-2  repeated failed action
-5  destructive/unsafe command
```

Keep the reward primarily tied to **verifiable outcomes** rather than subjective LLM judging.

For example:

```text
pytest -> all tests pass = strong positive reward
pytest -> tests fail      = little/no success reward
```

Start with a narrow task family so the reward is easy to interpret.

---

## 12. GRPO evaluation

Compare:

```text
Base Qwen3-1.7B
SFT + LoRA
SFT + DPO              (if used)
SFT + GRPO             (if used)
SFT + DPO + GRPO       (if used)
```

The exact sequence depends on the experiments. Do not force every method into the final pipeline.

Record:

- task success
- tool validity
- steps/task
- tokens/task
- reward/task
- latency
- failure modes

An ablation table is more valuable than listing every technique in the README.

---

## 13. Export the trained model for local inference

The laptop does **not** need the full BF16 Qwen weights for the final terminal agent.

The practical artifact is:

```text
fine-tuned adapter
       +
Qwen base
       ↓
merged/exported model
       ↓
4-bit GGUF
       ↓
llama.cpp
```

Use a Q4-family GGUF first.

Unsloth currently provides Qwen3-1.7B GGUFs, including Q4 variants around 1.1 GB, and documents running them through llama.cpp on Windows. This is well within the GTX 1060's 6 GB VRAM budget for inference, leaving room for runtime/KV-cache overhead. [Unsloth Qwen3-1.7B GGUF](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)

Your first local target should be:

```text
Q4_K_M or similar Q4 quantization
```

Then benchmark Q4 vs Q5 if quality justifies the additional memory.

---

## 14. Local terminal runtime

Recommended local stack:

```text
llama.cpp
    |
    | OpenAI-compatible local server
    v
127.0.0.1:PORT
    |
    v
Terminal agent harness
    |
    v
shell / files / grep / git
```

This gives you a clean separation:

```text
model server = inference
agent harness = reasoning loop + tools
executor      = actual computer actions
```

The terminal agent can therefore use a local model without requiring a large GPU.

---

## 15. Remote inference fallback

You should also support:

```text
terminal agent
      |
      +--> local llama.cpp server
      |
      +--> remote OpenAI-compatible endpoint
```

A remote endpoint is useful when:

- you test a larger model
- long context becomes expensive locally
- the fine-tuned model is not yet exported to GGUF
- you want to demo the project from another machine

The harness should use a generic model-client interface so swapping local/remote inference does not require changing the agent logic.

Example interface:

```python
class ModelClient:
    def generate(self, messages, tools):
        ...
```

Implement:

```text
LocalLlamaClient
RemoteOpenAICompatibleClient
```

---

## 16. Terminal agent safety model

For the first public version:

```text
workspace = temporary cloned repository
shell commands = sandboxed
command timeout = enforced
output = truncated
writes = logged
network = disabled by default
```

Do not let the public demo execute arbitrary commands against the user's entire home directory.

A strong portfolio project should demonstrate safe execution boundaries, not only model capability.

---

## 17. Final benchmark/demo

The final CLI should support something like:

```bash
term-agent "Fix the failing authentication tests"
```

Example flow:

```text
$ term-agent "Fix the failing authentication tests"

> pytest -q
  2 failed, 10 passed

> grep "hash_password" src/
  src/auth.py:14
  tests/test_auth.py:8

> read_file src/auth.py

> write_file src/auth.py

> pytest -q
  12 passed

Done. Fixed the authentication bug.
```

Log the complete trajectory for evaluation and debugging.

---

## 18. Portfolio README structure

### Title

**Terminal-SLM: Post-Training a Small Language Model for Reliable Tool Use**

### Sections

1. Problem
2. Why a 1.7B model?
3. Architecture
4. Tool protocol
5. Dataset construction
6. SFT + QLoRA
7. DPO / GRPO methodology
8. Executable evaluation environment
9. Benchmark results
10. Ablations
11. Quantization and local inference
12. Failure analysis
13. Safety boundaries
14. Demo
15. Reproduction instructions

### Include these charts/tables

- Base vs fine-tuned task success
- Tool-call validity
- Steps per successful task
- Tokens per task
- VRAM usage
- Latency
- Failure-category breakdown

---

## 19. Recommended execution order

### Phase A — Harness

1. Define tool schema.
2. Build shell/filesystem/grep/git tools.
3. Build the agent loop.
4. Connect base Qwen3-1.7B.
5. Make 10 tasks work manually.

### Phase B — Benchmark

6. Create 50-100 deterministic tasks.
7. Create the evaluator.
8. Establish the base-model baseline.

### Phase C — SFT

9. Create high-quality tool trajectories.
10. Run QLoRA SFT on a cloud GPU.
11. Evaluate.
12. Inspect failures.

### Phase D — Preference optimization

13. Turn real failures into preference pairs.
14. Run DPO if useful.
15. Re-evaluate.

### Phase E — RL

16. Convert the evaluator into a reward environment.
17. Start GRPO on a very small task set.
18. Validate that rewards correlate with real success.
19. Expand the benchmark.
20. Compare SFT/DPO/GRPO variants.

### Phase F — Local deployment

21. Export the trained model.
22. Quantize to GGUF.
23. Run llama.cpp locally.
24. Connect the terminal harness to the local server.
25. Benchmark local latency and VRAM.

### Phase G — Portfolio polish

26. Add failure analysis.
27. Add architecture diagram.
28. Add experiment tables.
29. Add reproducible training/evaluation commands.
30. Record a short terminal demo.

---

## 20. Minimum viable version vs full version

### MVP

```text
Qwen3-1.7B
+ QLoRA SFT
+ 5 terminal tools
+ 50 benchmark tasks
+ local GGUF inference
+ task-success evaluation
```

This is already a legitimate project.

### Full version

```text
Qwen3-1.7B
+ QLoRA SFT
+ DPO
+ GRPO
+ executable task environment
+ reward design
+ ablation study
+ local GGUF runtime
+ remote inference fallback
+ safety sandbox
+ public benchmark
```

Do not start with the full version. Build the MVP first and let the evaluation results determine whether DPO and GRPO are worth adding.

---

## 21. Final architecture

```text
                    USER
                     |
                     v
              terminal CLI
                     |
                     v
             AGENT HARNESS
          +----------+----------+
          |                     |
          v                     v
     MODEL CLIENT             TOOLS
          |               +-----+------+
          |               |            |
          v               v            v
   Qwen3-1.7B         shell/files   grep/git
   fine-tuned
      |
      v
  4-bit GGUF
      |
      v
   llama.cpp
      |
      +----------------------+
                             |
                             v
                        OBSERVATIONS
                             |
                             +-----> MODEL

Training / evaluation side:

benchmark task
     |
     v
agent trajectory
     |
     v
validator/tests
     |
     +------> metrics
     |
     +------> reward
              |
              v
             GRPO
              |
              v
       improved SLM
```

## 22. The central engineering principle

The project should answer one measurable question:

> **Can post-training make a 1.7B model reliable enough to operate a constrained terminal toolset on executable tasks?**

Everything else—LoRA, DPO, GRPO, the harness, quantization, and the benchmark—should exist to answer that question.
