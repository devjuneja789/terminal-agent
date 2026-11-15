# EXP-011 — Per-action predictions and label boundaries

Status: completed and audited; labels align, but critical file-tool choices remain wrong.

Job `6af87c5423985fb9cd823889`, HF T4 with 30-minute cap.
Config SHA-256: `2d3c0d3157bd3f642614b2e8ba01f842a376dbe50a860c721eff5c485fbd9573`.
All 97 local tests, Python compilation, shell syntax and diff checks passed.

## Question

EXP-010 reduced training loss and protocol violations but passed no tasks.
Determine whether the approved next actions have correctly aligned completion
labels and whether prediction errors concentrate on critical tool-name tokens,
arguments, or free generation. This is a read-only, support-only diagnostic.
No new training, tool execution, benchmark task changes or held-out access occur.

## Fixed scope and procedure

Use the same four approved references and 16 action targets, matching the
EXP-010 row hash. Pin its 64-step adapter commit
`270d688c111f3b01f72db6691d770e6f6cec024f`, subfolder
`20261114T145152Z/adapter`, and untouched base revision
`af89d219b53ed9b13b8a4645f9c8028973510324`. Use the same NF4, FP16 compute,
eager attention, seed 42, tokenizer and pinned runtime as EXP-010. Run on HF T4
with a 30-minute cap and a 1200-second inference budget. Each free generation
uses greedy decoding, at most 256 tokens and at most 60 seconds.

Reconstruct the exact TRL 1.7.1 dataset preprocessing/collator using a temporary
base model with untrained LoRA modules. Assert zero training steps and no
optimizer. Do not call train/backward. Discard this model before evaluating
fresh base/adapter inference models. Inspect each processed row by matching
its full prompt/completion contents to its approved source; never infer
identity from dataset order. TRL can shuffle the prepared dataset, while the
earlier mask audit labeled sources by positional index.

For all 16 targets, retain complete input IDs and labels, decoded labels,
rendered training text, the generation prompt IDs, the first labeled token,
mask contiguity, end-of-turn supervision, and exact comparisons between the
training prefix and inference prompt. Reproduce the prior mask-count multiset
and quantify incorrect source IDs in that earlier audit. Map the tool-name
character span to supervised tokens only when exact token-offset alignment
is verified.

For both base and adapter on each approved history:

- Teacher forcing: token accuracy, exact predicted target sequence, mean NLL,
  tool-name token correctness, first divergence, gold-token rank/probability,
  and top-five alternatives at incorrect/critical positions. Use the causal
  shift explicitly: logits at position j-1 predict label token j.
- Free generation: preserve raw text/token IDs, EOS/cap status, strict JSON
  and argument-schema validity, tool-name equality, and exact action-dictionary
  equality to the reference. No generated action is executed.

These later action prompts contain reference tool observations by design;
matching them measures prediction under gold history, not whether the model
can reach those histories in a live task. Exact finish text matching is a
memorization check, not a general semantic correctness score.

Save all 16 boundary records and 32 prediction records in bounded lossless
chunks. Extraction refuses incomplete archives. Record package versions,
GPU, row/config hashes, and source artifact identity. No new adapter or model
repository is created.

## Source inspection

TRL v1.7.1 prepares conversational prompts with add_generation_prompt=True,
labels the remaining suffix, and may shuffle the prepared dataset. Inspect
actual masks rather than assuming those boundaries are correct for this
tokenizer. Reference source:
https://github.com/huggingface/trl/blob/v1.7.1/trl/trainer/sft_trainer.py#L1387

## Results

Job `6af87c5423985fb9cd823889` completed, run `20261114T152447Z`. All 48
records were recovered: 16 boundary records and 32 prediction records. Raw
JSONL, metadata, summary, Markdown, source-log provenance and local consistency
analysis are under `research/results/tiny-aya-action-diagnostic/20261114T152447Z.*`.

### Boundaries and source identity

- All 16 training prefixes exactly match generation prompt token IDs.
- All 16 labeled suffixes decode to the exact approved action text.
- Labels are contiguous completion suffixes with no context labels; target
  labels equal input tokens at the same positions. Causal scoring uses j-1.
- Each action supervises one EOS/end-of-turn token; no missing stop label.
- Full training input sequences and token-offset mappings match the rendered
  templates in all cases. Tool-name spans are identified for all 16 targets.
- The prior mask-count multiset is reproduced. There are **zero** mismatched
  source IDs in the original audit for this run. The possible prepared-dataset
  shuffle issue was not realized here; do not label it a discovered source-ID
  bug. Diagnostic identity matching remains robust to shuffled data.

This audit provides no evidence of a masking or initial prompt-boundary fault
in these 16 examples. It does not certify every aspect of the training stack.

### Per-action prediction results

| Metric under approved reference histories | Base NF4 | 64-step adapter NF4 |
|---|---:|---:|
| Exact generated action dictionaries | 1/16 | 5/16 |
| Correct generated tool name (valid output) | 4/16 | 9/16 |
| Strict JSON/schema-valid generations | 7/16 | 15/16 |
| Exact teacher-forced target sequences | 0/16 | 5/16 |
| Teacher-forced tool-name token sequence correct | 8/16 | 9/16 |
| Mean per-action token accuracy | 86.48% | 91.35% |
| Mean per-action teacher-forced NLL | 0.91781 | 0.38899 |

These denominators are 16 gold-conditioned targets per arm, not the live
episode tool-call denominators in EXP-010. Later prompts include reference
observations; these are not independent repository-task outcomes.

| Expected tool | Targets | Adapter correct tool | Adapter exact action |
|---|---:|---:|---:|
| read_file | 4 | 1 | 0 |
| write_file | 3 | 0 | 0 |
| shell | 5 | 5 | 5 |
| finish | 4 | 3 | 0 |

The five exact matches are all named test-running shell commands. Exact finish
matching is deliberately literal; a paraphrase could be correct without
matching this reference. No semantic task score is inferred from that column.

### Critical decision tokens

For each write target, the first token of the tool name is `write`. The suffix
`_file` is near-certain once `write` is supplied by teacher forcing, but the
model chooses `shell` before it gets there.

| Intended write action | Adapter token accuracy | P(write), rank | Top token |
|---|---:|---:|---|
| B021 page-size edit | 96.75% | 0.3014, rank 2 | shell |
| B026 environment handling edit | 97.55% | 0.0443, rank 5 | shell |
| B041 default-value edit | 98.67% | 0.0935, rank 4 | shell |

The B021 P(write) increased from 0.0843 in the base to 0.3014 in the adapter,
but did not overtake shell. Thus weights and some choices changed, while
greedy decoding still selects the wrong action. High accuracy over copied
file contents and JSON tokens can coexist with this decisive tool-name error.

Tool names are not the only remaining errors. Under the correct teacher-forced
write context, B021 predicts `min` where the patch needs `max` and `0` where
it needs `1`; B026 predicts the default-return pattern where `int(...)` is
required. All three write targets also have a closing-brace token error.
B041's content tokens, including the new default value, are predicted correctly;
its two wrong tokens are the tool-name decision and final brace structure.
This localizes why mostly correct copied content does not yield an edit action.

Three of four read targets likewise choose shell at the `read` token. Only
the B046 read picks read_file, and it selects a different line range from the
gold reference. Free-generation edit proposals are test commands or an
inappropriate `git config -f shop/config.py ...` command instead of write_file.
They were recorded, never executed.

## Conclusion and next experiment

The model learned the five test-running shell actions and improved structured
output. It has not learned when to select the file-reading/writing tools.
The observed failure is not explained by absent labels for these actions or
a differing initial inference prefix. More global token accuracy is an
insufficient acceptance criterion for this project.

A controlled decision/edit-focused objective is the next hypothesis to test:
give explicit weight to the supervised tool-name decision, changed code spans
and required JSON closure,
while retaining full completion/EOS supervision. Keep the approved data and
paired replay fixed, and monitor per-action tool choice and executable edits.
This is a proposed experiment, not an established remedy. No extra training,
data expansion, tool execution or held-out evaluation was performed here.
