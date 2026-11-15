# Release review

This directory is the submission package under preparation. No file here
asserts that an artifact or submission has already been made public.

The public source URL is expected to be
https://github.com/devjuneja789/terminal-agent . Local uncommitted work must be
included before a public link is considered reproducible. The recorded demo is
standalone `demo.html`, not a live terminal endpoint.

## Concrete publication scope

1. Publish the final code, experiment records, configs, immutable review
   provenance, real results, write-up and walkthrough. Exclude virtualenvs,
   local model weights, credentials/config caches, scratch `my-project`,
   `.terminal-slm`, `.agents`, `.codex`, `.aws`, bytecode and verbose job logs.
2. Add model cards to the versioned experimental adapter repositories. Publish
   the project's own adapters only; do not redistribute the gated base model.
   Retain Tiny Aya attribution, its noncommercial license and acceptable-use
   requirements. Pin the original artifact commit/subfolder even when a card
   update creates a newer repository commit.
3. Publish the explicit `support-data` payload as
   `devjuneja789/terminal-slm-reviewed-support`. It contains four admitted
   demonstrations, 15 targets and review/admission provenance. No rejected or
   held-out target becomes training data.
4. Publish `evaluation-logs` as
   `devjuneja789/terminal-slm-evaluation-evidence`. Held-out traces are visibly
   evaluation-only. Include model/base failures and AuditKIT coverage limits.
5. Create two public HF collections: models/adapters, and data/evaluation.
   Check all links without authentication before submitting the form.

`scripts/prepare_submission_release.py` prepares these payloads and a file
checksum manifest after the final evidence index exists. It performs no
network calls or public writes. Review its output before publication.

## Owner decisions and final form

The remaining owner review concerns the public release scope and the project's
own source/data license. Base-model derivative terms are already retained and
cannot be replaced with a more permissive license. The form also needs the
registered identity/email, final public collection URLs and project abstract.
No account credentials are needed in chat.

The official deadline remains Nov 15, 2026, 23:59 AoE (Nov 16, 11:59 UTC);
keep the user's earlier midnight cutoff. Source:
https://hackathon.lexsi.ai/#submit . A functioning submission does not imply
that the judges will accept negative task-success results as meaningful gain.
