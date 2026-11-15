# Manual Hugging Face publication steps

The focused, checksummed upload payload is staged at:

```text
submission/hf-upload
```

It contains two adapter cards, approved support data, the frozen held-out and
EXP-027 evidence, and the fixed benchmark. It omits rejected/pending support
candidates and older diagnostic runs. It does not contain base-model weights.

## 1. Verify authentication and payload

In the WSL terminal where you will upload:

```sh
cd /mnt/e/terminal-agent
.venv/bin/hf auth whoami
python3 - <<'PY'
import json
from pathlib import Path
p=Path('submission/hf-upload/release-manifest.json')
m=json.loads(p.read_text())
print(m['status'], len(m['files']), 'files')
PY
```

If `whoami` is not the intended HF account, authenticate with
`.venv/bin/hf auth login` in that terminal. Never paste the token into chat or a
tracked file.

## 2. Publish the two adapters

These adapter repositories already contain the uploaded weights. In each HF
repository's **Settings**, change visibility to Public. Then upload the
corresponding model card from the focused payload:

```sh
.venv/bin/hf upload devjuneja789/terminal-slm-tiny-aya-joint-300-v1 \
  submission/hf-upload/models/README-terminal-slm-tiny-aya-joint-300-v1.md \
  README.md --type model

.venv/bin/hf upload devjuneja789/terminal-slm-tiny-aya-task-start-620-v1 \
  submission/hf-upload/models/README-terminal-slm-tiny-aya-task-start-620-v1.md \
  README.md --type model
```

The first adapter is the EXP-025 held-out candidate. The second is an EXP-027
support-only diagnostic; its card explicitly says it was not evaluated on
held-out tasks. Do not upload or redistribute the base-model weights. Add the
upstream Tiny Aya repo only as a linked item in the collection.

## 3. Create and upload the datasets

Create these two dataset repositories on Hugging Face, selecting **Public**:

- `devjuneja789/terminal-slm-reviewed-support`
- `devjuneja789/terminal-slm-evaluation-evidence`

Then upload the staged folders:

```sh
.venv/bin/hf upload devjuneja789/terminal-slm-reviewed-support \
  submission/hf-upload/support-data . --type dataset

.venv/bin/hf upload devjuneja789/terminal-slm-evaluation-evidence \
  submission/hf-upload/evaluation-logs . --type dataset
```

The support repository's license metadata is CC-BY-NC-4.0. Its training files
contain only approved targets. The evaluation repository labels held-out
records evaluation-only; never use them as training data.

## 4. Create the two collections

From your Hugging Face profile, create a public collection titled
**Terminal-SLM models and adapters** and add:

1. `CohereLabs/tiny-aya-global` (link only; do not re-upload its weights)
2. `devjuneja789/terminal-slm-tiny-aya-joint-300-v1`
3. `devjuneja789/terminal-slm-tiny-aya-task-start-620-v1`

Create a second public collection titled **Terminal-SLM data and evaluation**
and add the two dataset repositories above. Copy the generated collection URLs
into the hackathon form; collection URLs include Hub-generated identifiers, so
do not guess them.

## 5. Publish the article and verify

Open <https://huggingface.co/new-blog>, paste `submission/HF-ARTICLE.md`, and
replace its three URL placeholders with the actual public GitHub and collection
links. Publish only after those links work in a logged-out/private browser.
The account must be eligible to publish a personal Hub article; if the editor
blocks publication, use the account eligibility guidance in the Hub docs or
ask the hackathon organizers what alternative article URL they accept.

The final form needs four real URLs: model collection, dataset collection,
published article, and GitHub repository. Do not submit placeholders or links to
private repositories.

## Repair the evaluation dataset viewer

The repository mixes tasks, training history and nested model traces. The
viewer must use explicit configurations instead of automatically combining
all JSON/JSONL files. Rebuild the viewer copies and corrected card locally:

```sh
python3 scripts/prepare_hf_evidence_viewer.py
hf upload devjuneja789/terminal-slm-evaluation-evidence \
  submission/hf-upload/evaluation-logs . --type dataset
```

This uploads the corrected root README, nine display JSONL files in `viewer/`,
and `viewer-manifest.json`, alongside unchanged evidence. Each config has an
explicit string schema, with JSON-encoded display cells for variable nested
structures. Original evidence files are preserved. After the Hub rebuilds the
viewer, select `final-heldout-runs` (20 rows) and the other configurations from
the dropdown; `task-start-training-history` should show 621 rows. The old
inferred default config is replaced by the named configs. If any new build
fails, capture its configuration name and error text for diagnosis; do not
change benchmark outcomes or suppress failed runs.
