# Anonymous release checklist

Complete this checklist against the exact folder or archive that will be
submitted.

- [ ] Publish this folder from a fresh repository, not from the development
      repository or its history.
- [ ] Follow `ANONYMOUS_4OPEN_SCIENCE.md`: use a dedicated private source
      repository, neutral Git identity, pinned reviewed commit, auto-update
      off, and **Remove when expired**.
- [ ] Include only the files listed by `find . -type f`; do not add parent
      folders, checkpoints, caches, virtual environments, logs, results,
      unlisted datasets, scheduler scripts, or symlinks. The one expected data
      manifest is `data/qwen36_offline_dataset/manifest.json`.
- [ ] Do not include API configuration files. Provide credentials only through
      environment variables at runtime.
- [ ] Keep public model IDs in configs. Inspect every generated
      `adapter_config.json`, model card, tokenizer config, and safetensors
      metadata for serialized cache paths before sharing.
- [ ] Run `bash scripts/audit_anonymity.sh` and a dedicated secret scanner on
      the staged release.
- [ ] Test installation and unit tests from a clean environment without the
      parent source tree on `PYTHONPATH`.
- [ ] Run `scripts/reconstruct_dataset.py --verify-only` and confirm all five
      dataset parts remain at or below 7,000,000 bytes.
- [ ] Resolve the dataset redistribution license documented in
      `data/DATASET_CARD.md`; preserve the Stanford Alpaca and cleaned-Alpaca
      attributions and keep the data license separate from the code license.
- [ ] Manually review or institutionally accept the heuristic PII-like content
      flags documented in `data/DATASET_CARD.md`. Do not describe the corpus as
      free of personal information based only on automated scans.
- [ ] Add institutionally approved code and dataset licenses before public
      upload. This anonymous review copy intentionally makes no licensing
      grant.
- [ ] Configure Anonymous GitHub redaction terms narrowly. Dataset parts are
      binary and are served without textual redaction, so all content checks
      must be completed before pushing.
- [ ] Open the anonymous URL while logged out, download its ZIP, rerun the
      reconstruction/content/anonymity audits, and verify the raw SHA-256 is
      `cd349baa406eae0bf7e4ee2d938d087d6656469d88076d68ba76baeb7e5ec669`.
- [ ] Build any archive without `.git`, extended attributes, or absolute
      symlinks. Normalize stored owner/group names, numeric IDs, and timestamps
      (or use `git archive` from the fresh anonymous repository), inspect the
      archive headers, extract into a new temporary directory, and rerun all
      scans.
- [ ] Do not add the submission PDF to this repository. Anonymous GitHub does
      not redact binary PDF contents or metadata.
