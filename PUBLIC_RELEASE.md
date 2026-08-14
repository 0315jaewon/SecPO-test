# Publishing the public SecPO repository

This directory is the identity-bearing public source snapshot for
<https://github.com/Sizhe-Chen/SecPO>. It was built as a fresh, reviewed tree;
do not merge the research workspace or its history into it.

## Publication gates

Before publishing a commit:

1. Confirm that every relevant copyright holder and institution authorizes
   distribution of the SecPO-owned code and documentation under Apache-2.0.
   The repository license does not license datasets, model weights,
   benchmarks, external patches, or third-party components.
2. Resolve the preference-dataset redistribution decision in
   `data/DATASET_CARD.md`. A code license does not license the dataset.
3. Resolve the provenance and license compatibility of the adaptive-attack
   implementations and retain every required third-party notice and citation.
   Apache-2.0 cannot grant rights the SecPO copyright holders do not own.
4. Verify that `CITATION.cff` and `CITATION.bib` contain the agreed five-author
   citation: Sizhe Chen, Yibo Peng, Jaewon Chang, Chawin Sitawarin, and David
   Wagner.
5. Run the model-free tests, exact allowlist validator, and public-release
   audit from a clean checkout.
6. Run a dedicated secret scanner on the staged tree and Git history.

## Fresh-history workflow

Create the public repository from this folder alone. Never filter, mirror, or
copy the development repository's history: deleted credentials, private paths,
large model artifacts, and identifying infrastructure may remain recoverable.

```bash
bash scripts/audit_public_release.sh
git init
git switch -c main
git config user.name "Sizhe Chen"
git add --all
git commit -m "Release SecPO training and evaluation code"
git remote add origin https://github.com/Sizhe-Chen/SecPO.git
git push --set-upstream origin main
```

Review `git status --short`, `git diff --cached`, and
`git log --format=fuller --max-count=1` before pushing. Do not put credentials
in a remote URL or command line. Use the GitHub credential manager, GitHub App,
SSH agent, or `gh auth login` instead.

## Model weights and runtime state

This repository intentionally contains no base models, adapters, merged
weights, checkpoints, endpoint values, credentials, logs, or results. Publish
large weights through a model registry that supports sharded artifacts and
model cards, then link immutable revisions from this repository. Inspect every
model/tokenizer/PEFT metadata file for local cache paths before uploading.

Runtime provider settings belong in environment variables or an untracked
secret manager. The checked-in code must work with injected local callables or
OpenAI-compatible routing contracts without embedding a provider key or
institutional endpoint.

## Verification

From the repository root:

```bash
python -m pytest -q
python -m compileall -q src scripts tests examples
bash -n scripts/*.sh
python scripts/validate_release_manifest.py
bash scripts/audit_public_release.sh
```

Build release archives with `git archive` from the reviewed public commit.
Extract the archive into a new temporary directory and rerun the manifest,
dataset, and security audits against the exact downloadable artifact.
