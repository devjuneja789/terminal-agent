# EXP-024: Final fixed fit-budget test and candidate freeze

## Protocol frozen before execution

The user approved the submission-day plan. One final optimization-adequacy
test, then candidate freeze and submission preparation. Same 15 admitted
support-only targets and immutable row hash, same fresh Tiny Aya base and joint
objective as EXP-023, same optimizer, LR, seed, quantization, AMP scale 256 and
all-batch gradient preflight. Change only training horizon from 60/four epochs
to **300 updates/20 epochs**, with the existing linear schedule stretched over
that horizon. No checkpoint sweep, new examples, model/prompt changes or
adaptive extension. HF T4-small training cap 40 minutes.

After audited training/reload, evaluate the same 15 exact contexts on base and
candidate with the existing native-stop procedure. Require 45 complete records,
training-token/weight reproduction and independent action/stop/control audits.
No tools execute in the diagnostic. Inference cap 30 minutes.

Before any new held-out evaluation, compare the completed joint-60 and new
joint-300 adapters on these support contexts using lexicographic score:
exact complete action matches, valid finish choices, schema-valid outputs,
valid correct tool choices. Prefer joint-60 on a tie. If the new run fails,
retain joint-60 with a recorded failure reason. Freeze the chosen immutable
artifact in `configs/submission_candidate.json`; the freezer refuses overwrite.
No held-out information enters selection. No more training after this test.

The selected candidate then receives a fresh matched held-out comparison under
one versioned native-stop/NF4 configuration. Historical v1 baseline results
are not silently reused as this new runtime's baseline. Submission assets and
AuditKIT reporting are prepared during GPU execution. No positive result is
assumed and no final benchmark task may change.

## Status

Training completed on HF T4-small as Job `6af955ae29dd05ff03685217`, submitted
by `scripts/run_tiny_aya_fit300.sh`. Run `20261115T064937Z` completed all 300
updates/20 epochs in 1038.677 training seconds. Average train loss 0.319914.
All fixed-profile, mask, joint-loss, all-batch gradient, update and reload
checks pass. Artifact repo `devjuneja789/terminal-slm-tiny-aya-joint-300-v1`,
commit `1eed40612640c78d88d9fe0cac36e06b1148d1fd`, subfolder
`20261115T064937Z/adapter`. No held-out task was accessed.

The fixed exact-context comparison completed as HF Job
`6af95a9523985fb9cd830399`, run `20261115T071048Z`. All 45 records and lineage,
training-token/weight reproduction, action-score and stopping checks pass.

| Metric on 15 reviewed contexts | Untouched base | Joint-60 | Joint-300 |
|---|---:|---:|---:|
| Valid correct tool choice | 1/15 | 8/15 | 15/15 |
| Exact complete action | 1/15 | 3/15 | 6/15 |
| Schema-valid output | 15/15 | 14/15 | 15/15 |
| Correct finish choice | 0/4 | 0/4 | 4/4 |
| Correct read choice | 0/2 | 0/2 | 2/2 |
| Correct write choice | 0/3 | 2/3 | 3/3 |
| Exact file-write action | 0/3 | 0/3 | 2/3 |
| Mean target NLL | 0.810573 | 0.754039 | 0.282947 |

Zero token-cap hits. Joint-300 wins the preregistered tuple comparison
`(6,4,15,15) > (3,0,14,8)` and is selected. The immutable candidate freeze was
written **2026-11-15 07:14:40 UTC**, before final held-out staging/evaluation.
No further training is planned. These are reviewed training-context predictions,
not held-out or executable task-success results. Finish contents still differ
from the gold answers; two shell arguments, both read ranges and one edit also
remain inexact. Correct tool choice alone does not establish completion.

Inspection of the four finish predictions finds useful paraphrases for the
three repair contexts, including the lower/upper-bound fix and default 15.
The B026 answer is terse. B046 correctly mentions the observed 10-versus-25
mismatch but misleadingly attributes it to a configuration file missing the
environment setting, rather than the implementation ignoring the environment.
Thus 4/4 finish choices is not a 4/4 factual-answer score. These outputs are
diagnostic predictions, not new approved demonstrations or training targets.

The full local suite passes 163 tests (four optional tests skipped under
standard-library Python); all seven focused checks pass with the actual
AuditKIT SDK installed. EXP-025 now tests the frozen candidate end-to-end.
