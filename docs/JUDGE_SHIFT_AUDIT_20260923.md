# Offline audit: GPT-4o-mini versus GPT-5.6 Luna utility scores

Audited September 23, 2026. No new generations or judge API requests were made.
Original evaluation artifacts were read-only throughout the audit.

## Main finding

The AlpacaEval2 decline reflects changed preferences on the same answer pairs,
not just a change in probability weighting or length-control arithmetic. SEP
also has substantial judge disagreement, but gains offset losses much more
closely. In both benchmarks, moving to Luna raises scores for candidates shorter
than their references and lowers scores for longer candidates, on average.
The different balance and magnitude of those effects explains the aggregate
contrast descriptively; length alone has not been established as its cause.

## Scope and results

Compared the recommended-decoding, non-thinking Lambda generations in
`outputs/qwen38_lambda_recommended_20260916` against their Luna rejudgments in
`outputs/qwen38_lambda_luna_openai_20260918`. Each model has 805 AlpacaEval2
examples and 9,160 SEP rows. The lr4e-5 model has no corresponding mini result
in this experiment, so it is not part of the paired comparison.

Scores below are percentages; changes are percentage points.

| Model | Alpaca raw: mini → Luna | Change | Alpaca LC: mini → Luna | Change | SEP raw: mini → Luna | Change |
|---|---:|---:|---:|---:|---:|---:|
| Base | 89.28 → 71.03 | -18.26 | 79.91 → 64.81 | -15.10 | 85.41 → 88.60 | +3.19 |
| SecPO lr8e-5 | 90.76 → 73.70 | -17.06 | 77.51 → 64.50 | -13.02 | 90.50 → 88.71 | -1.79 |
| SecPO lr6e-5 | 90.45 → 71.50 | -18.95 | 79.65 → 61.93 | -17.72 | 89.38 → 88.31 | -1.07 |

Correction to the earlier explanation: length control **attenuates**, rather
than amplifies, the judge-switch decline for all three models. For the base,
the raw drop is 18.26 points and the LC drop is 15.10 points. SEP's primary
metric is raw probability-weighted win rate, not LC win rate, but comparing raw
against raw still shows the cross-benchmark contrast.

## What was verified

The audit script verifies the following for all six model/benchmark pairs:

- Candidate-file checksums agree between judge runs; current saved candidates
  also match those checksums. Each paired annotation has identical instruction,
  candidate response, reference response, and generator identities.
- Reference-file checksums and evaluator-package source hashes match the run
  records. SEP uses the user-approved alternate reference file, consistently
  for both judges; it is not silently substituted with the release reference.
- Original judge prompts are reconstructed using the actual AlpacaEval 0.6.6
  annotator and its seeded answer-order randomization. Every reconstructed
  request is satisfied from its existing content-addressed cache, with no API
  fallback. Both judges receive the same rendered message sets, including the
  same answer order for each pair.
- Replaying the saved choices through the parser and answer-order correction
  reproduces all saved preferences, with maximum error 0.
- Recomputing raw, discrete, and (where applicable) length-controlled scores
  reproduces every saved metric to numerical tolerance.

This rules out mismatched response files, a changed comparison prompt, a
between-run answer-order mismatch, or an error in reaggregating these saved
judgments as explanations. It is not a proof that either judge is correct.

Both runs use the probability-weighted `logprob_parser`: the candidate score
is its normalized verdict-token probability among `m` and `M`, after restoring
the original response order. These token scores are not calibrated human
preference probabilities. Luna was **not** converted to hard labels for these
completed direct-OpenAI runs. For the base Alpaca set, even a hard preference
score drops from 89.69% to 71.37%, almost the same drop as weighted scoring.

Luna often returns only one verdict alternative in `top_logprobs`, despite
requesting five. The audit bounds the possible effect of the unreported verdict
mass on the raw mean at less than 0.12 percentage points for every model/suite;
this cannot explain a 17–19 point raw Alpaca decline. The bound uses
`u / (v + u)`, where `u` is all unreported token probability and `v` is the
observed valid-verdict probability, and assumes the returned probabilities are
accurate. It does not bound broader judge calibration differences or the LC fit.

## SEP's stable aggregate hides disagreements

Here a win means candidate token score > 0.5; a loss means < 0.5. The table
omits transitions involving exact ties. These are preferred-answer counts,
not counts of probabilities that round to 0 or 1.

| Model | Alpaca mini win → Luna loss | Alpaca reverse | SEP mini win → Luna loss | SEP reverse |
|---|---:|---:|---:|---:|
| Base | 195 | 47 | 588 | 892 |
| SecPO lr8e-5 | 180 | 40 | 766 | 597 |
| SecPO lr6e-5 | 196 | 37 | 731 | 653 |

Thus 30.1% of base Alpaca pairs and 16.2% of base SEP pairs flip direction.
SEP is not judge-invariant: its opposing changes largely cancel.

## Length-associated changes

These are raw probability-weighted scores for the **base** model, grouped by
candidate/reference **character** length ratio. Character length is also what
the installed LC implementation uses. Bins were chosen for interpretation,
not to recreate the LC regression.

| Candidate/reference length | Alpaca n | Alpaca mini → Luna | Change | SEP n | SEP mini → Luna | Change |
|---|---:|---:|---:|---:|---:|---:|
| ≤1× | 97 | 42.13 → 74.60 | +32.47 | 2,464 | 54.92 → 81.02 | +26.10 |
| >1–2× | 422 | 95.88 → 74.47 | -21.40 | 4,036 | 95.56 → 91.88 | -3.68 |
| >2–4× | 209 | 96.81 → 68.54 | -28.27 | 2,291 | 98.36 → 91.12 | -7.24 |
| >4× | 77 | 92.10 → 54.35 | -37.74 | 369 | 97.51 → 87.67 | -9.84 |

The shorter-or-equal group is 26.9% of SEP but only 12.0% of Alpaca. Its gains
contribute +7.02 points to the overall SEP mean, outweighing -3.83 points from
longer responses. On Alpaca those contributions are +3.91 and -22.17 points.
Both SecPO variants show the same direction in every length bin: shorter
candidates gain and longer candidates lose under Luna. Their per-bin values
are saved in the machine-readable summaries.

This supports a judge difference associated with verbosity, but does not
isolate length bias: factuality, relevance, task type, and reference quality
can all covary with answer length. An LC fit cannot remove substantive errors
that happen to occur in longer responses, and does not guarantee elimination
of every task-dependent length effect.

Within base Alpaca, the largest component decline is `helpful_base`:
98.14% → 58.61% across 129 examples, with 52 win-to-loss reversals and no
loss-to-win reversals. All five source components decline, so the result is
not limited to a single component.

## Inspected examples and limits of qualitative interpretation

Indices are zero-based annotation indices. These are illustrative cases found
by inspecting early disagreements, **not a blinded or representative manual
quality study**. Full pairs are preserved in the disagreement files.

- **Alpaca index 17, badminton explanation:** the candidate says the opponent
  can return the shuttle before it hits the ground twice, then says it must be
  hit before touching the ground. It also says volleying is forbidden. Mini
  gives the candidate approximately 1; Luna gives it 0. The ground-twice claim
  is incorrect: [BWF law 15.2](https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/SECTION%204.1-%20Laws%20of%20Badminton.pdf)
  says the shuttle is out of play once it hits the court. The reference avoids
  that error. This is a concrete quality issue, not just a length difference;
  it does not establish that every statement in the reference is correct.
- **SEP index 20, Polish marketing translation:** the candidate supplies a
  concise Polish translation. The reference supplies a translation followed
  by a long English explanation of its wording. Mini gives the candidate
  approximately 0; Luna gives it 1. This illustrates a gain for a concise
  candidate without claiming that the translation's quality was independently
  evaluated by a Polish speaker.
- **SEP index 38, upcoming art events:** no location or dates are provided.
  The candidate explicitly says specific upcoming events cannot be identified
  from the input. The reference instead supplies hypothetical event names.
  Mini gives the candidate approximately 0; Luna gives it 1. This is consistent
  with a preference for grounding over elaboration.
- **SEP index 54, extracting agreement names:** both outputs list the Paris
  Agreement and the United Nations Framework Convention on Climate Change.
  Mini gives the candidate 0.6225; Luna gives it 0. This is a useful warning
  against interpreting every confident Luna reversal as a clear quality win.

The judges emit only verdict tokens, not explanations. Their reasons cannot
be recovered from these files. The examples support a hypothesis about
factuality, grounding, and verbosity; they do not prove Luna's rationale or
that Luna is universally more accurate.

## Important protocol distinctions and remaining uncertainties

Alpaca here compares against **GPT-4o-2024-05-13 reference responses**, not the
GPT-4-Turbo references described in the paper summary. SEP compares against
**Llama-3-8B-Instruct responses**. These references are unchanged across our
two judge runs, so they do not invalidate the within-benchmark comparison.
But the different opponents and tasks help explain why the same judge can
have different aggregate effects across benchmarks. Reference strength and
task composition cannot be disentangled using these two datasets alone.

The judge transports also differ: mini used OpenRouter, which returned a mix
of OpenAI and Azure providers; Luna used direct OpenAI. Mini requested
`max_tokens=1`; Luna requested `max_completion_tokens=16` with
`reasoning_effort=none`. Both requested temperature 1 and logprobs, and Luna's
validator requires exactly one completed verdict token. These are not
identical model/endpoint configurations, even though prompts and scoring are
held fixed. Align these details when comparing to a colleague's Luna results.

Because the candidate answers were reused, changed generation sampling,
output caps, or truncation **between judge runs** cannot cause this shift.
Thinking-on could change the candidate answers and therefore the judge gap,
but the existing non-thinking experiment does not establish that it will.
Also, the base model loses substantially too: this judge-switch decline is
not evidence by itself of a new fine-tuning-induced utility regression.

## Useful follow-up

First align with the colleague on reference-file checksum, metric/parser,
judge prompt, model name, endpoint, reasoning setting, and answer ordering.
Then generate thinking-on responses once and judge the **same** saved outputs
with both judges, using the same references. A blinded human review of a
prespecified sample of disagreements would test the quality hypothesis more
directly. None of those new evaluations was launched by this audit.

## Reproduce and inspect

Run from the repository root in the existing evaluation environment:

```bash
runtime/eval_envs/lm/bin/python -u scripts/analyze_lambda_judge_shift.py
```

This script needs the existing local artifacts and installed pinned evaluator;
it is not a standalone downloadable evaluation bundle. A missing cached request
raises an error instead of contacting an API.

Reports are stored on REDS at:

`/data2/jaewonc/secpo_analysis/judge_shift_20260923/`

- `summary.json`: six comparisons, audit-script hash, metrics, length groups,
  task components, cache identities, and replay checks.
- `{base,secpo,secpo_lr6e5}/{alpacaeval2,sep_utility}/summary.json`: individual
  comparison summaries.
- The same directories' `paired_examples.csv`: all per-example scores and
  lengths, aligned across judges.
- The same directories' `disagreements.json`: full instruction/reference/
  candidate text for examples whose preferred-answer labels differ.

The report files include benchmark text and model outputs, but no API keys.
