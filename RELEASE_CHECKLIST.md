# Public release checklist

Complete this checklist against the exact commit and archive that will be
published.

- [ ] Verify that `CITATION.cff` and `CITATION.bib` contain the agreed
      five-author citation: Sizhe Chen, Yibo Peng, Jaewon Chang, Chawin
      Sitawarin, and David Wagner.
- [ ] Confirm that all relevant copyright holders and institutions authorize
      distribution of the SecPO-owned code and documentation under
      Apache-2.0. Verify that this license is not presented as licensing
      datasets, models, benchmarks, external patches, or other third-party
      components.
- [ ] Resolve the separate dataset redistribution license and documented
      content-review gates in `data/DATASET_CARD.md`.
- [ ] Publish from fresh Git history containing only this folder. Do not reuse,
      mirror, or filter the research repository's history.
- [ ] Validate the exact tree with
      `python scripts/validate_release_manifest.py`. Every non-Git file must be
      exactly allowlisted; global denials remain in force.
- [ ] Run `bash scripts/audit_public_release.sh` and a dedicated secret scanner
      against both the staged files and reachable Git objects.
- [ ] Confirm there are no credentials, private endpoints, machine paths,
      scheduler directives, logs, results, runtime exchanges, caches, model
      weights, checkpoints, PDFs, notebooks, archives, or symlinks.
- [ ] Inspect every released model ID, adapter config, model card, tokenizer
      config, and weight manifest for serialized local cache paths. Weights are
      distributed separately and are never added here.
- [ ] Test installation and unit tests in a clean environment without the
      parent research workspace on `PYTHONPATH`.
- [ ] Run `scripts/reconstruct_dataset.py --verify-only`; confirm the raw
      Qwen preference checksum and all five hosting-safe part checksums.
- [ ] Preserve Stanford Alpaca and cleaned-Alpaca attribution, and keep any
      dataset license distinct from the project code license.
- [ ] Re-run the AgentDojo, AgentDyn, and DTap checkout verifiers against fresh
      clones; confirm immutable commits, upstream licenses, patches/templates,
      and absence of unrelated modifications.
- [ ] Resolve the origin and license compatibility of the Genetic/MAP-Elites,
      PAIR, and TAP implementations; retain all required notices and citations.
      Do not treat the project Apache-2.0 license as a grant of third-party
      rights.
- [ ] Run model-free smoke tests for Llama and Qwen Genetic/PAIR/TAP, Qwen
      reasoning normalization, exact nine-vote SEP judging, the full
      AgentDojo paper-80 manifest, and interrupted/resumed search.
- [ ] For online Llama training, verify that the attack worker observes the
      current ready-marked policy snapshot during PAIR search and that the
      defender/attacker exchange is isolated by run ID and fingerprint.
- [ ] For AlpacaEval2 and SEP Utility, validate the external reference files by
      row count, generator identity, and SHA-256 before generation or judging.
- [ ] If releasing merged Qwen or Llama weights separately, rerun the merge
      from a clean environment, scan generated metadata, shard large files,
      publish hashes, and document the base-model and adapter licenses.
- [ ] Build the source archive with `git archive`, inspect its headers and file
      list, extract it into a fresh directory, and rerun all audits there.
- [ ] Do not include the submission PDF, proprietary prompts/provider clients,
      GCG, baselines, ablations, instruction-hierarchy experiments, temporary
      scripts, checkpoints, or generated experimental state.
