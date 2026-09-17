# Triage Classifier — Validation Notes

## Objective
Check whether a small, interpretable model can reproduce (and eventually
generalize beyond) the hand-written IMCI-style danger-sign rules used in
the USSD/SMS triage prototype, before any real patient data exists.

## Data
20,000 synthetic cases, split across three caller types: child under
5 (40%), adult (45%), pregnant woman (15% — this mix itself is still
an unsourced assumption). Per-symptom prevalence is a mix of sourced
national figures and explicitly-flagged placeholders:

| Feature | Value | Status | Source / caveat |
|---|---:|---|---|
| child: fever (proxy for `fever_gt3d`) | 23% | Sourced (proxy) | NDHS 2023–24, any fever in prior 2 weeks — duration not broken out nationally |
| child: ARI symptoms (`breathing`) | 2.6% | Sourced (proxy) | NDHS 2018 — mother-reported ARI symptoms, 2 weeks |
| child: diarrhea (`diarrhea_dehydration`) | 9% | Sourced (proxy) | NDHS 2023–24, any diarrhea — dehydration severity not isolated |
| child: danger signs | 2% | **Unsourced placeholder** | No population-representative rate found for IMCI danger-sign frequency at point of call |
| adult: cough ≥2wk | 5% | **Unsourced placeholder** | Only source found (26.6%) was a self-selected TB-outreach sample, not general population — would badly overstate the rate |
| adult: danger signs | 2% | **Unsourced placeholder** | No source found |
| pregnant: swelling+headache | 4.5% | Sourced | Nigeria-wide meta-analysis, pooled preeclampsia prevalence |
| pregnant: danger signs | 1.4% | Sourced (floor only) | Same meta-analysis, eclampsia component — excludes bleeding/reduced fetal movement, so true rate is higher |

Resulting class balance (imbalance is expected and realistic, not a
data bug):

| Tier   | Share |
|--------|-------|
| Green  | 82.7% |
| Yellow | 15.3% |
| Red    | 2.0%  |

## Why raw accuracy is the wrong metric here
On the *illustrative* data, a default `DecisionTreeClassifier`
(max_depth=4, no class weighting) reached 95.9% accuracy but caught
only **14.8% of Red cases** — it had enough depth budget to split on
`breathing` (10% prevalence) and `swelling_headache` (12%), but not
enough left over to also isolate `danger` in every branch, so most
Red cases fell through into a Yellow or Green leaf. That failure was
invisible in the accuracy number.

## Re-checking this after sourcing the data — the finding changed
Rerunning the identical script on the sourced-prevalence data
produced a different result: **the unweighted model now also gets
100% Red recall**, matching the class-balanced version exactly (both:
98.5% accuracy, Red 1.000, Yellow 0.905, Green 1.000 at depth=4).
Inspecting the actual tree confirmed why — `breathing` and
`swelling_headache` are now rare enough (2.6%, 4.5%) that the tree
doesn't spend depth on them, which leaves room for `danger` to appear
in every branch at depth=4, fully separating Red without any
reweighting needed.

This is reported instead of quietly updated, because it changes the
conclusion: **class weighting isn't a fix that reliably works — it's
a check that has to be run every time**, since whether a shallow tree
catches the rare class depends on how it competes for depth against
every other feature's prevalence, not on class weighting alone. A
submission that kept the original "unweighted models miss emergencies"
claim after this data update would be citing a result the current
data no longer produces.

## Where class weighting still matters
| Model                                   | Red recall | Yellow recall | Green recall |
|-------------------------------------------|-----------:|--------------:|-------------:|
| Unweighted, depth=4 (sourced data)         | 1.000      | 0.905          | 1.000        |
| `class_weight='balanced'`, depth=4         | 1.000      | 0.905          | 1.000        |
| `class_weight='balanced'`, depth=6         | **1.000**  | **1.000**      | **1.000**    |

Depth is still what fixes the remaining gap — at depth=4, 73 of 766
true Yellow cases (9.5%) fall through to Green because there isn't
room to split on every relevant feature. Depth=6 recovers full recall
on all three tiers. The practical rule to carry forward isn't
"always use class_weight='balanced'" — it's **always check per-class
recall on the rare classes specifically, at whatever depth you land
on, because neither weighting nor depth alone guarantees it.**

## Final model
- `DecisionTreeClassifier(max_depth=6, class_weight='balanced')`
- **100% agreement** with the hand-written rules across all 20,000
  synthetic cases (0 disagreements)
- Fully readable — the learned tree can be printed as plain if/else
  rules and checked branch-by-branch against the clinical logic

## Limitations (stated plainly, not hidden)
- **This validates internal logical consistency, not real-world
  accuracy.** The model agrees with the rules it was trained to
  imitate — it has not been checked against any real patient outcome.
- Metrics were re-checked after swapping in sourced prevalence and
  held (100% recall on all tiers, 100% rule-agreement) — but that's
  expected: the model is still learning the same 6-feature branching
  structure regardless of how realistic the input distribution is.
  Sourcing the data changes *what population it's representative of*,
  not whether the model can learn the rules.
- Two feature groups (child and adult danger-sign frequency) remain
  unsourced placeholders — no published Nigerian rate exists for how
  often a caller reports an IMCI-style danger sign. This is the
  single most important number to replace once real session data is
  available, since it directly drives Red-tier prevalence.
- No free-text or voice input is modeled — only structured yes/no
  answers, matching the USSD interface.

## Next step: real data
Once the system is live, every session logged (phone, ward, symptoms,
tier assigned) plus a CHW's confirmation of the true outcome becomes a
real labeled example. Retraining periodically on that data — and
specifically re-checking Red recall each time — is what turns this
from "a model that agrees with our assumptions" into "a model
validated against real cases."
