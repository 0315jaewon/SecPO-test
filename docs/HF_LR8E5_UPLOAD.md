# Original lr=8e-5 model publication

Destination: `promptinjections/Qwen3.8-27B-SecPO-lr8e-5` (private).

This publishes the complete merged BF16 model, not an adapter or optimizer
checkpoint. The existing REDS backup is the source of truth:

```text
/data2/jaewonc/secpo_backups/lambda_20260919/models/secpo_lr8e5
```

## Launch and resume

From `/home/jaewonc/SecPO`, with write access to the destination organization:

```bash
runtime/eval_envs/lm/bin/python scripts/upload_hf_lr8e5.py
```

Enter the Hugging Face token at the hidden prompt. It is passed through a pipe
to a detached worker, not stored in a credential file, command line, or model
artifact. The original evaluation credential file is not changed. Run this only
when a previous upload worker is no longer active; a file lock rejects concurrent
workers. To resume after an interruption, rerun the command and provide a valid
token. Upload metadata remains available locally for resumption.

The job verifies all 27 source files against the preserved backup SHA-256
manifest, then stages them with hard links (no second 55.6 GB weight copy).
It adds a model card, the original base license, sanitized training provenance,
the original merge validation report, and `SHA256SUMS`. An explicit file allowlist
excludes all caches, credentials, datasets, training examples, and logs.

The destination must be private. Existing files must have matching contents;
the script refuses unexpected or conflicting repository contents.

## Status

```bash
less /data2/jaewonc/secpo_hf_uploads/qwen38_secpo_lr8e5/status.json
tail -n 40 /data2/jaewonc/secpo_hf_uploads/qwen38_secpo_lr8e5/upload.log
```

`phase: uploading` does not mean the repository is complete. The final
`phase: complete` is written only after all remote filenames and sizes match,
every large-file SHA-256 matches, and every small Git blob checksum matches at
one pinned repository revision. The status records that revision.

Revoke any token that was shared in chat after the upload completes. The
repository and uploaded model remain available after token revocation.

## Offline checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 runtime/eval_envs/lm/bin/python -m pytest -q tests/test_hf_lr8e5_upload.py
```

These exercise remote-file validation using mock metadata; the real upload also
performs the full source and remote integrity checks described above.
