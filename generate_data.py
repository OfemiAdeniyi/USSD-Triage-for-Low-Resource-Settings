"""
Generate synthetic training data for the triage classifier.

Prevalence sources, mapped feature-by-feature. Where no population-
representative Nigerian estimate exists, that is stated explicitly
rather than silently kept as a guess -- these are the fields to
prioritize once real session data starts coming in.
"""
import numpy as np
import pandas as pd
from rules import label_case

rng = np.random.default_rng(42)
N = 20000

who_probs = {"child": 0.40, "adult": 0.45, "pregnant": 0.15}

PREV = {
    "child": {
        # SOURCED: NDHS 2023-24 -- 23% of under-5s had ANY fever in
        # prior 2 weeks (national report doesn't break out >3-day
        # duration specifically; this is a proxy, not an exact match).
        "fever_gt3d": 0.23,
        # SOURCED: NDHS 2018 (via Statista) -- 2.6% of under-5s had
        # ARI symptoms (short/rapid or difficult chest breathing)
        # nationally in prior 2 weeks.
        "breathing": 0.026,
        # SOURCED: NDHS 2023-24 -- 9% of under-5s had diarrhea in
        # prior 2 weeks (national figure doesn't isolate the subset
        # with dehydration signs specifically; proxy, not exact).
        "diarrhea_dehydration": 0.09,
        # UNSOURCED -- no population-representative Nigerian estimate
        # found for danger-sign prevalence (convulsions, unable to
        # drink/breastfeed, etc.) in this age group. Kept as a
        # conservative placeholder pending clinical/programmatic input
        # or real session data.
        "danger": 0.02,
    },
    "adult": {
        # UNSOURCED -- population-level "cough >2 weeks" prevalence
        # was only found from a self-selected TB-outreach sample
        # (26.6%, Lagos slum survey) which would badly overstate the
        # general-population rate. Kept as a conservative placeholder
        # until a better source (e.g. WHO/NTBLCP survey microdata) is
        # available.
        "cough_fever_gt2wk": 0.05,
        # UNSOURCED -- no population-representative estimate found.
        "danger": 0.02,
    },
    "pregnant": {
        # SOURCED: Nigeria-wide pooled meta-analysis -- preeclampsia
        # prevalence 4.51% (95% CI 3.82-5.29%) among pregnant women.
        "swelling_headache": 0.045,
        # SOURCED (partial): pooled eclampsia prevalence 1.39%
        # (95% CI 1.02-1.84%) is used as a FLOOR for "danger" -- it
        # only captures convulsions, not bleeding or reduced fetal
        # movement, so the true prevalence of ANY danger sign is
        # almost certainly higher than this.
        "danger": 0.014,
    },
}

ALL_FEATURES = ["danger", "breathing", "diarrhea_dehydration", "fever_gt3d",
                "cough_fever_gt2wk", "swelling_headache"]

rows = []
whos = rng.choice(list(who_probs.keys()), size=N, p=list(who_probs.values()))

for who in whos:
    symptoms = {f: 0 for f in ALL_FEATURES}
    for feat, p in PREV[who].items():
        symptoms[feat] = int(rng.random() < p)
    label = label_case(who, **symptoms)
    row = {"who": who, **symptoms, "label": label}
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("synthetic_triage_data.csv", index=False)

print(df["label"].value_counts(normalize=True).round(4))
print("\nRows:", len(df))
