# EXP-013: Can the weighted adapter preserve a supplied complete file?

## Question and fixed procedure

EXP-012 produced schema-valid writes that replaced required module code. B021
and B041 saw only source headers; B026 saw the complete module but did not write.
Can the weighted adapter produce a preserved, correct file after the exact
approved full-source observations, without further training?

Use the EXP-012 adapter at commit
`1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0`, subfolder
`20261114T160511Z/adapter`, with the existing EXP-011 action diagnostic. Reuse
the same four references/16 action targets and exact token sequences, NF4/FP16
inference, greedy decoding, 256-token cap and package pins. Score fresh base
and weighted adapter; compare the archived unweighted adapter from EXP-011.
No optimizer, backward, new demonstrations or held-out records are allowed.

The primary cases are the approved write targets B021/B026/B041. Each approved
prefix already contains a full original-file read. Other action predictions
remain available as context for tool-choice and formatting errors. The 256-token
cap exceeds each approved write's target length; cap hits are still reported.

Compare the three weighted live first decisions after source reads with their
approved-history decisions. Live B021/B041 observations are partial; live B026
is full. Histories differ in more than source length, so this comparison cannot
isolate source length causally. No benchmark or instruction is modified.

## Preservation and executable checks

Report write selection, exact gold content, formatting-insensitive gold AST
equality, and missing/altered original top-level nodes. Unchanged imports,
assignments, docstring and definitions must retain their AST; original node
identities must remain present. Preservation alone does not establish a fix.

For schema-valid writes to the expected path, apply only that generated content
to a disposable pinned task snapshot using the existing write_file tool. Run
the task's unchanged named validator; never execute a generated shell command.
Retain stdout/stderr, changed paths and validation status. This conditional
single-action test is not an agent task-success score. Preserve source hashes,
approved data hashes and all raw predictions. No output becomes training data.

## Status

### Additional regression check, identified after inspecting predictions

The generated B021 body uses `max(value, 1)`, omitting the previous upper cap.
Run the snapshot's existing `test_page_size_upper_bound` separately on that
write. This is a post hoc diagnostic check, not a new benchmark task, changed
acceptance command or altered success score. Keep its outcome separate from
the frozen named lower-bound validation. No benchmark file is edited.

GPU predictions and local preservation/validation audit complete.
Launcher: `bash scripts/run_tiny_aya_source_preservation.sh`.

## Configuration and provenance

- HF T4 job `6af89f5229dd05ff036796d4`, diagnostic run `20261114T175228Z`.
- Config SHA-256: `61abf773bfa08475a6c23ef3305bc64c8a9df1300ede349c80578fa26f55e894`.
- Unchanged approved action-row SHA-256:
  `ef2ef134f4b30dc822879e85d238e206c1c6ef9e949d83f49c60c21c75df33ae`.
- All 16 input/label sequences match EXP-011 exactly. All prefixes, completion
  labels, offsets and EOS checks pass. Fresh-base aggregate metrics reproduce
  EXP-011 exactly; no optimizer or training call ran.
- All 16 boundary and 32 prediction records were recovered losslessly. Results,
  metadata, source/log hashes and independent analysis are under
  `research/results/tiny-aya-source-preservation/20261114T175228Z.*`.
- Local analyzer verified approved original source against the pinned snapshot,
  unchanged task contracts, and failing pristine validators for all three cases.
  It applied four eligible predicted writes (two approved-context and two
  archived live-context writes) only to disposable copies. Each copy was removed.

## Results

### Primary edit decisions with approved complete source

| Task | Selected action | Original surrounding code preserved | Exact approved content | Named validator | Additional finding |
|---|---|---|---|---|---|
| B021 | write_file | Yes | No | PASS | Existing upper-bound test FAILS: 101 != 100 |
| B026 | read_file | No write | No | Not run on a prediction | Requests lines 32–42 of the supplied 18-line file |
| B041 | write_file | Yes | Yes | PASS | Complete module matches gold exactly |

The weighted adapter selects write_file on 2/3 edit decisions; both writes
retain all unchanged original top-level nodes and definitions. Neither hits
the token cap; all three primary predictions terminate with EOS. B041's entire
JSON action and file content match the approved reference, changing only the
default timeout to 15. This establishes that this adapter can reproduce at
least one complete correct edit under an approved source/history condition.

B021 uses `max(value, 1)` instead of preserving the existing upper cap with
`max(1, min(value, MAX_PAGE_SIZE))`. It retains the surrounding module but
changes too much behavior inside the edited function. The frozen named test
checks only zero input, so it passes; the existing upper-bound regression test
fails. Report both results. No benchmark, score or validator was rewritten.

B026 remains an action-choice failure despite complete source and prior failing
test evidence. Under the approved target prefix, the first `write` token has
probability 0.2514 (rank 3), behind read and shell. B021 and B041's write-token
probabilities are 0.8082 and 0.6823, both rank 1. High conditional token accuracy
(B026: 97.55%) still masks a critical decision error.

### Comparison with archived live decisions and controls

| Condition | Write selected / 3 | Surrounding module preserved / 3 | Exact gold file / 3 | Named validator passes / 3 |
|---|---:|---:|---:|---:|
| Fresh base, approved histories | 0 | 0 | 0 | 0 (no writes validated) |
| Unweighted 64-step, approved histories (EXP-011) | 0 | 0 | 0 | 0 (no writes validated) |
| Weighted 64-step, approved histories | 2 | 2 | 1 | 2 |
| Weighted live first decisions after reads (EXP-012) | 2 | 0 | 0 | 0 |

Live B021/B041 first read only 4/2 source lines and replace the module with
snippets, producing NameErrors. Approved histories supply the full file before
the prediction, and their generated writes preserve the surrounding module.
This supports investigating source gathering, but different preceding actions,
read arguments and observation formatting prevent attributing the difference
solely to source length. B026 is a useful counterexample: its live read already
contained all 18 lines, yet neither history produces a write.

### All-action diagnostic context

| Metric, 16 approved action prefixes | Fresh base | Unweighted 64-step | Weighted 64-step |
|---|---:|---:|---:|
| Exact free-generated actions | 1 | 5 | 6 |
| Correct free-generated tool names | 4 | 9 | 14 |
| Schema-valid free generations, including finish | 7 | 15 | 15 |
| Teacher-forced exact sequences | 0 | 5 | 6 |
| Mean action token accuracy | 86.48% | 91.35% | 92.20% |
| Mean action NLL | 0.91781 | 0.38899 | 0.39253 |

The weighted adapter's six exact actions are the five existing shell-test
actions plus the B041 complete-file write. These are conditional next-action
predictions, not live task completions. Its end-to-end replay remains EXP-012's
0/4. No held-out success or generalization gain is established.

## Conclusion and next step

The adapter can preserve a supplied complete module on the two cases where it
chooses to write, and can exactly reproduce the approved B041 edit. It is not
reliably correct: B021 introduces a behavior regression, and B026 still chooses
the wrong action. Treat file preservation, edit correctness and source-gathering
behavior as separate failures rather than extending training based on loss alone.

Before more training, test a controlled full-source observation intervention on
the same support contexts, applying any general harness change uniformly to
base and adapter in a separately versioned replay. Keep the original evaluation
and results frozen. Preserve existing behavior checks and verify failed-test
claims against observations. No intervention, new training, dataset expansion
or held-out run was launched in this experiment. No diagnostic output is an
approved training target.

Verification: 114 local tests, three optional Torch skips. The eight focused
preservation tests pass, including isolation, lineage rejection, formatting
tolerance, snippet rejection and the lower-bound-pass/upper-bound-fail case.
