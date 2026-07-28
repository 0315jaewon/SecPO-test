# Publishing through Anonymous GitHub

This artifact is prepared for the public Anonymous GitHub instance at
<https://anonymous.4open.science/>. The service creates an anonymized view of
a GitHub repository; it is not itself the source repository.

Do not upload until the separate code-license, dataset-license, and
PII-like-content decisions in `RELEASE_CHECKLIST.md` and
`data/DATASET_CARD.md` are complete. Anonymous access is still public
redistribution.

## 1. Create a dedicated source repository

Use a new **private** GitHub repository containing only this folder's
contents. Do not push the parent development repository, its history, or a
subdirectory of it. A public source repository can be found through ordinary
GitHub search even when the proxy link is anonymous.

Prefer a GitHub account that has no unrelated private repositories. GitHub
OAuth does not offer read-only source-code access to private repositories, so
authorizing an OAuth application for private-repository access has a broader
permission surface than this task needs.

Before the first commit:

```bash
bash scripts/audit_anonymity.sh
git init
git switch -c anonymous-release
git config user.name "Anonymous Authors"
git config user.email "<reserved-nonidentifying-address>"
git add --all
git commit -m "Anonymous review artifact"
git status --short
git log --format=fuller --max-count=1
```

Use a neutral private-repository name and push the reviewed snapshot:

```bash
git remote add origin <private-repository-url>
git push --set-upstream origin anonymous-release
```

Do not use Git LFS for this artifact. Anonymous GitHub's public-service FAQ
states that files above 8 MB are unsupported, while binary files are served
without content anonymization. The audited dataset is therefore a
deterministic gzip stream split into 7,000,000-byte normal Git objects. Every
part remains below the service limit, and the manifest reconstructs and
verifies the exact experimental JSON. Anonymous GitHub does not document an
end-to-end LFS guarantee.

## 2. Create the anonymized view

Sign in at <https://anonymous.4open.science/> and create a new anonymization:

1. Paste the dedicated private GitHub repository URL.
2. Select `anonymous-release` and pin the exact reviewed commit SHA.
3. Turn **Auto update off** for the final submission snapshot.
4. Enter the conference ID if the conference supplied one.
5. Choose a random, neutral repository ID. Do not use an author, institution,
   internal project, or source-repository name.
6. Add one redaction expression per line for every author name, username,
   email address, institution, organization, private repository name, and
   machine/storage prefix that could identify the authors. These entries are
   regular expressions, so escape punctuation unless regex behavior is
   intended.
The dataset parts are binary and will not be redacted by the service; audit
them locally before pushing.
7. Keep links enabled only after confirming every retained link is public and
   non-identifying. This repository's documented links are to public upstream
   projects and service documentation.
8. Leave image, PDF, notebook, and GitHub Pages display disabled; this artifact
   does not require them.
9. Select **Remove when expired**, not redirect, so expiration cannot reveal
   the source GitHub repository. Set the date after the complete review,
   rebuttal, and decision period, or use the conference-provided policy.

The service may take several minutes to prepare a repository of this size.

## 3. Verify as a reviewer

Open the resulting `/r/<id>/` URL in a logged-out private browser window.
Check all of the following:

- The source GitHub owner, repository name, commit author, and email are not
  visible.
- README rendering, source browsing, and repository ZIP download work.
- No redaction rule changed code, model IDs, file names, or dataset contents.
- The downloaded dataset matches its committed checksum.

After downloading and extracting the anonymous snapshot:

```bash
python scripts/reconstruct_dataset.py \
  --manifest data/qwen36_offline_dataset/manifest.json \
  --output /tmp/qwen36_offline_preferences.json
python scripts/audit_release_dataset.py \
  /tmp/qwen36_offline_preferences.json \
  --schema raw \
  --expected-records 19157
bash scripts/audit_anonymity.sh
```

Only place the Anonymous GitHub URL—not the source GitHub URL—in the paper.
Do not place `secpo_submission.pdf` in this repository: PDF content and
metadata are binary and are not anonymized as text by the service.

## 4. Updating the artifact

For a correction, rerun every audit, create a new neutral commit, push it, and
manually update the anonymization to that reviewed SHA. Recheck the logged-out
view and downloaded checksum. Avoid automatic branch tracking after the paper
has been submitted.

After review is complete, follow the conference's archival policy, remove the
Anonymous GitHub authorization if no longer needed, and publish a normal
identity-bearing repository only when doing so no longer compromises review.

## References

- [Anonymous GitHub service](https://anonymous.4open.science/)
- [Anonymous GitHub source and operating model](https://github.com/tdurieux/anonymous_github)
- [Anonymous GitHub public-service FAQ](https://github.com/tdurieux/anonymous_github/blob/main/public/partials/faq.htm)
- [GitHub regular-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub OAuth authorization guidance](https://docs.github.com/en/apps/oauth-apps/using-oauth-apps/authorizing-oauth-apps)
