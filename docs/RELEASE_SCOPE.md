# Release and backup scope

`release_manifest.json` is the source of truth for what may enter this public
artifact. It exists because the research workspace contains much more than the
paper-main pipeline, and because the desired public or backup scope may change.

The manifest is descriptive and the validator is read-only. It does not copy,
delete, stage, archive, upload, or back up anything.

## Decision vocabulary

Every audited component has exactly one classification:

- `include`: clean material present in the release or approved for it.
- `refactor`: scientifically relevant code that must be made portable,
  focused, security-reviewed, and testable before inclusion.
- `external`: a separately distributed model, dataset, framework, or
  benchmark that must be installed at a pinned immutable revision.
- `exclude`: baselines, ablations, follow-on studies, generated state, or
  one-time operations outside the main artifact.
- `missing`: a paper-main component not available in the audited source tree.

Component decisions explain scientific scope. The active profile separately
enumerates every literal release file. This separation lets maintainers
discuss a planned component without accidentally allowing its old
research-workspace files.

## Profiles and later changes

`public_release` is the active profile. `maintainer_source_backup` currently
inherits that reviewed scope and adds nothing. It is intentionally a
placeholder for later decisions about what source to retain.

Both profiles use `allow_mode: exact`. Wildcards are rejected, every listed
path must exist, and every non-Git file in the candidate tree must be listed.
This makes an accidentally copied script a validation failure even when its
extension and parent directory look ordinary.

To change the scope:

1. Move or add a component decision with a concise reason and release gate.
2. Add each reviewed file by its exact relative path; do not replace the exact
   allowlist with a directory or extension wildcard.
3. Add essential new files to `required_paths`.
4. Resolve licenses, provenance, identity leakage, machine paths, endpoints,
   and generated metadata before changing `refactor` to `include`.
5. Add focused tests and run the manifest and public-release audits.
6. Review the exact Git diff and create a new reviewed snapshot commit.

Changing an allow rule never overrides a global deny. Widening or removing a
deny rule requires a separate security review. A maintainer-only archival
backup that legitimately contains credentials, private infrastructure, raw
logs, or model artifacts must use a different private process and must not be
derived by weakening this public-release manifest.

## Non-negotiable denials

The manifest rejects credential/key files, embedded credential material,
private or machine-local endpoints and paths, identity-bearing email
addresses, symlinks, scheduler directives, caches, environments, logs,
results, outputs, checkpoints, model weights, PDFs, notebooks, and files not
covered by the active allowlist. Dataset archive chunks have a narrow size
exception and remain subject to their checksum and content audit.

The validator reports only a rule label and file path for content findings; it
does not print matching secret text.

The DTap compatibility patch has one exact-path exception for loopback-only
benchmark wiring. Those addresses connect processes on the evaluator host and
cannot identify an institution. The exception does not suppress private
network ranges, private DNS suffixes, machine paths, credentials, or any other
content rule, and wildcard exception paths are rejected.

Two additional DTap patch matches are pinned by rule name and SHA-256: a
synthetic benchmark account address and its synthetic task-local access token.
The validator accepts only those exact matched strings at that exact path and
fails if either exception becomes stale. This does not permit a different
address, token, credential assignment, or match elsewhere in the tree.

## Validation

From the release root:

```bash
python scripts/validate_release_manifest.py
python scripts/validate_release_manifest.py --schema-only
python -m pytest -q
bash scripts/audit_public_release.sh
```

Run validation against the clean candidate snapshot, not a working directory
after tests have produced cache files. All failures are release blockers until
reviewed; do not silently add a broad allow rule merely to make validation
green.

## History boundary

The public repository must start from fresh history containing only the
reviewed tree. Never preserve, filter, squash, or mirror the parent research
history: deleted credentials, private infrastructure, large files, and local
paths can remain recoverable. Use the approved public author metadata and
inspect the staged tree and resulting archive again.
