# Pre-Registration: _Confidently Wrong: Group-Conditional Miscalibration in Accent Classifiers_

**Author:** Mihnea Popescu
**Date committed:** April 30th 2026
**Git tag:** `prereg-v1`

This document is committed _before_ any test-set evaluation. It fixes the
baseline model, datasets, splits, metrics, hard-case predictions, and the
decision rule for whether the intervention is considered successful. Any
deviation from this document in the final paper will be reported explicitly
in a "Deviations from Pre-Registration" section.

---

## 1. Research question

Do off-the-shelf accent classifiers produce confidence scores that are
equally well-calibrated across speakers from different first-language (L1)
backgrounds? If not, can group-conditional temperature scaling close the
calibration gap, and at what cost?

---

## 2. Hypotheses

**H1 (disparity exists).** Per-group Expected Calibration Error (ECE) will
differ by at least a factor of 2 between the best-calibrated and
worst-calibrated group on the baseline model.

**H2 (intervention helps).** Group-conditional temperature scaling will
reduce worst-group ECE by at least 30% relative, compared to the unmodified
baseline.

**H3 (single-T is insufficient).** A single global temperature value, fit on
the pooled calibration set, will reduce average ECE but will not reduce
worst-group ECE by more than 15% relative — meaning the disparity persists
under naive calibration.

---

## 3. Baseline model

- **Baseline:** `Accent-ID-CommonAccent-ECAPA`
  (16-class English accent classifier, ECAPA-TDNN architecture).
  - Source: `Jzuluaga/accent-id-commonaccent_ecapa` on HuggingFace.
  - Version pin: 14bebf44b7e7a34204d0acc2c897935945fb5c51

The 16 CommonAccent labels are: african, australia, bermuda, canada, england
hongkong, indian, ireland, malaysia, newzealand, philippines, scotland
singapore, southatlandtic, us, wales

---

## 4. Dataset

**`edinburghcstr/edacc`** (Edinburgh International Accents of English
Corpus). 40 hours of dyadic English conversation, ICASSP 2023 release
(Sanabria et al., arXiv:2303.18110), CC-BY-SA license, available on
HuggingFace.

**Why EdAcc and not Common Voice:**

- EdAcc was released _after_ the model's training, so contamination risk
  is zero.
- EdAcc is conversational rather than read speech, more closely matching
  real deployment.
- EdAcc provides per-speaker L1 metadata, enabling cleaner per-group
  analysis than CV's self-reported accent field.

**Audio handling:** EdAcc clips will be segmented into single-speaker
utterances using the dataset's speaker-turn annotations. Utterances under
3 seconds will be flagged but not excluded (their behavior is one of the
hard-case predictions). Audio will be resampled to 16 kHz mono on load.

---

## 5. Splits

EdAcc speakers will be split with a fixed seed (`SEED = 42`) at the
**speaker level** (not utterance level — to prevent the same speaker
appearing in both calibration and test):

- **Calibration split:** 50% of speakers per L1 group, used for fitting
  temperature scaling parameters.
- **Test split:** 50% of speakers per L1 group, used for all reported
  metrics. Untouched until Phase 1 evaluation.

A "qualitative held-out" set will be drawn by randomly selecting 8–10
specific utterances from the test split _after_ quantitative analysis is
complete, for the qualitative analysis section. These utterances will be
listened to manually.

Speaker-stratified splits will be saved as CSVs in `splits/` and committed.

---

## 6. Group definitions and the in-vocab / OOV split

**A group is defined by the speaker's self-reported L1.**

Two analysis tracks:

**Track A — In-vocabulary L1s.** L1s that map cleanly to one of the model's
16 CV labels via the manual mapping documented in
`paper/l1_to_cv_mapping.md`. The mapping will be committed _before_ test-set
evaluation. Provisional mapping (subject to confirmation against the actual
EdAcc L1 distribution and the model's label list):

| EdAcc L1 / accent                                  | CV label    | Notes            |
| -------------------------------------------------- | ----------- | ---------------- |
| English (raised in US)                             | us          | direct           |
| English (raised in England)                        | england     | direct           |
| English (raised in Scotland)                       | scotland    | direct           |
| English (raised in Ireland)                        | ireland     | direct           |
| English (raised in Australia)                      | australia   | direct           |
| English (raised in Canada)                         | canada      | direct           |
| English (raised in New Zealand)                    | newzealand  | direct           |
| Hindi / Tamil / Bengali / Urdu / other South Asian | indian      | grouped          |
| English (raised in Singapore)                      | singapore   | direct           |
| English (raised in Malaysia)                       | malaysia    | direct           |
| English (raised in Philippines)                    | philippines | direct           |
| English (raised in Hong Kong)                      | hongkong    | direct           |
| Yoruba / Igbo / Swahili / other African            | african     | grouped — coarse |

**Track B — Out-of-vocabulary L1s.** L1s with no equivalent in the model's
label space. Provisional list (subject to EdAcc's actual L1 distribution):
Mandarin, Cantonese, Japanese, Korean, Polish, Russian, German, Spanish,
French, Arabic, Portuguese, Vietnamese, Thai, Turkish, Italian.

**Minimum group size for inclusion:** 4 speakers per L1, contributing at
least 100 utterances total. Groups below threshold will be excluded from
per-group metrics and listed in an "Excluded Groups" appendix.

---

## 7. Metrics

**Track A (in-vocabulary, where ground-truth label exists):**

1. Per-group Expected Calibration Error (15 equal-width bins). Headline metric.
2. Per-group reliability diagrams (small multiples).
3. Per-group top-1 accuracy.
4. Per-group selective accuracy at 80% coverage.
5. Per-group Brier score.

**Track B (out-of-vocabulary, no ground-truth label):**

1. Per-group mean top-1 confidence.
2. Per-group prediction entropy distribution.
3. Difference between in-vocab mean confidence and OOV mean confidence —
   the "knows-it-doesn't-know" gap. Smaller absolute difference = worse
   epistemic calibration.

ECE will be computed from a hand-written implementation, unit-tested against
a synthetic case, and cross-checked against `torchmetrics.CalibrationError`.

---

## 8. Pre-registered hard cases

Three concrete predictions, committed _before_ running any evaluation:

**HC1 (in-vocab disparity, expected to hold).** Among in-vocabulary L1
groups, "indian" (which aggregates many South Asian L1s) and "african"
(which aggregates many African L1s) will exhibit the highest ECE — at
least double that of "us" and "england". This reflects coarse label
aggregation hiding within-group heterogeneity.

**HC2 (OOV overconfidence, expected to hold).** Mean top-1 confidence on
OOV speakers will be within 10 percentage points of mean top-1 confidence
on in-vocab speakers — meaning the model does not appropriately reduce
confidence when the speaker is outside its label space.

**HC3 (utterance length, expected to hold).** Within any L1 group, ECE on
utterances under 3 seconds will be at least 30% higher than ECE on
utterances over 5 seconds — meaning the model is systematically
overconfident on inputs too short to identify reliably.

After Phase 1 evaluation, each prediction will be reported as
**held / partially held / failed** with the actual numbers. Predictions
that fail will be discussed honestly.

---

## 9. Intervention

**Primary intervention:** Group-conditional temperature scaling, fit on the
EdAcc calibration split. For each in-vocab group `g`, fit a single scalar
`T_g` by minimizing NLL via PyTorch LBFGS (50 iterations, init `T = 1.0`).

**Comparison points:**

- Unmodified baseline (`T = 1.0`).
- Single global `T` fit on the pooled in-vocab calibration set.

**OOV speakers:** group-conditional scaling cannot directly help here
(no group-specific T can be fit without ground-truth labels). We will
report the effect of (a) the global T and (b) an "average in-vocab T"
applied to OOV speakers, and note that neither is a real solution. This
is the trust-tax discussion's headline finding.

**Stretch interventions** (only if Phase 2 finishes ahead of schedule):

1. Conformal prediction with per-group calibration; report set sizes
   per group at 95% marginal coverage.
2. Entropy-threshold abstention; report coverage-accuracy curves.

LoRA fine-tuning is **not** in scope for this revised plan.

---

## 10. Decision rule

The intervention is considered **successful** if both:

1. Worst in-vocab group ECE is reduced by at least 30% relative versus
   unmodified baseline, AND
2. Best in-vocab group ECE does not increase by more than 20% relative.

**Partially successful** if exactly one criterion is met. **Failed** if
neither is met.

The OOV overconfidence finding (H2) is treated as descriptive, not as a
success criterion — its purpose is to ground the trust-tax discussion.

Per the assignment, well-analyzed negative results receive full credit.
The paper will be written honestly regardless of outcome.

---

## 11. Deviations policy

Any change to this document after the `prereg-v1` git tag will be:

1. Committed as a new tag (`prereg-v2`, etc.) with a clear commit message.
2. Reported in a "Deviations from Pre-Registration" section in the paper.
3. Justified with a reason (e.g., dataset unavailability, bug discovered).

Cosmetic edits (typos, formatting) do not require a new tag.

---

## 12. What this document does _not_ cover

- Final paper structure and section ordering (flexible).
- Specific figure aesthetics.
- Related work selection.
- Phrasing of trust-tax discussion.

These are write-up decisions and are not subject to pre-registration.
