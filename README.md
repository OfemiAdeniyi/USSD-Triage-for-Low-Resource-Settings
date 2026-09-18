# USSD/SMS Triage for Primary Health Centers

A "Small AI" triage system built for the World Bank Global AI & Digital
Summit 2026 Hackathon (Health track). It lets a caller check symptoms
over a basic feature phone — no smartphone, no app, no internet
connection required — and, for urgent cases, alerts the nearest health
post with the caller's number, ward, and a transport recommendation.

**Live demo:** tested via Africa's Talking's USSD sandbox simulator
(see [Limitations](#limitations) for why real-phone dialing isn't
possible at this stage).

---

## Why this exists

Most triage and health-tech tools assume a smartphone and a data
connection — both unreliable or unavailable for a large share of
rural primary health center (PHC) catchments in Nigeria. USSD and SMS
work on any phone, on any network, with no data plan. This project
asks: **can a small, interpretable triage model run entirely over
that channel, and still be trustworthy enough to act on?**

## How it works

1. A caller dials a USSD code. The Flask backend walks them through:
   **ward → who the check is for (child / adult / pregnant woman) →
   a short sequence of yes/no danger-sign questions**, matching a
   simplified IMCI-style decision tree.
2. The moment a "Yes" answer determines a tier (Red/Yellow/Green), the
   session ends with clear guidance for the caller.
3. **Red cases** additionally get: an SMS alert toward the relevant
   health post/rider network with the caller's number and ward, and a
   transport recommendation (ambulance if the ward has one, nearest
   rider network if not).
4. Every session is logged, and a **small trained decision-tree model
   runs alongside the rules as a live cross-check** — not as the
   decision-maker. If the model and the rules ever disagree, it's
   logged for review. The rules stay authoritative because they're
   deterministic and auditable; the model's agreement rate becomes
   real validation data over time.

## Architecture

```
Caller's phone
   |  USSD session
   v
Africa's Talking (gateway)
   |  POST /ussd
   v
Flask app (app.py) — hosted on Render
   |-- rules-based decision tree (authoritative)
   |-- trained decision-tree model (triage_tree_final.joblib) -- cross-check only
   |-- SQLite (sessions.db) -- session log
   `-- Africa's Talking SMS API -- Red-case alerting
```

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | The Flask USSD/SMS backend — the actual deployed service |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes `venv/`, `sessions.db`, `.env` from version control |
| `rules.py` | Ground-truth triage rules (the "teacher" used to label synthetic training data) |
| `generate_data.py` | Generates the synthetic training dataset, using a mix of sourced Nigerian prevalence rates and explicitly-flagged placeholders |
| `train_model.py` | Trains and compares an unweighted vs. class-balanced decision tree |
| `finalize_model.py` | Trains and saves the final model (`triage_tree_final.joblib`) and its feature order (`feature_columns.csv`) |
| `synthetic_triage_data.csv` | The generated training data |
| `triage_tree_final.joblib` | The trained model, loaded by `app.py` at runtime |
| `feature_columns.csv` | Exact feature column order the model expects — required alongside the model file |
| `REPORT.md` | Full validation write-up: data sources, class-imbalance findings, model performance, and stated limitations |

## Running it locally

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python app.py
```

This starts the Flask app on `http://localhost:5000`. To let Africa's
Talking's sandbox reach it, expose it with
[ngrok](https://ngrok.com):

```bash
ngrok http 5000
```

Then set your Africa's Talking USSD channel's Callback URL to
`https://<your-ngrok-url>/ussd`.

### Environment variables

None of these are required to run the triage flow itself — only for
real SMS sending:

| Variable | Purpose |
|---|---|
| `AT_USERNAME` | Africa's Talking username (`sandbox` for testing) |
| `AT_API_KEY` | Your Africa's Talking sandbox API key |
| `AT_ALERT_RECIPIENT` | Phone number that receives Red-case SMS alerts |

If these aren't set, the app still runs correctly — it just logs what
it *would have* sent instead of sending it.

## Deployment

Currently deployed on [Render](https://render.com)'s free tier.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`
- Environment variables set under Render's "Environment" tab (see
  table above)

## The model

Full methodology, data sources, and findings are in
[`REPORT.md`](./REPORT.md). Summary:

- Trained on 20,000 synthetic cases, generated from Nigerian
  prevalence data where published rates exist (NDHS 2018/2023–24 for
  child fever/diarrhea/ARI; a Nigeria-wide meta-analysis for
  pregnancy danger signs), and explicitly flagged placeholders where
  they don't (child/adult danger-sign frequency).
- A `DecisionTreeClassifier` (max_depth=6, `class_weight='balanced'`)
  reaches 100% recall on Red, Yellow, and Green tiers, and 100%
  agreement with the hand-written rules.
- An earlier finding — that an *unweighted* model badly under-caught
  Red cases — did not reproduce once the prevalence data was sourced.
  `REPORT.md` explains why, and states the corrected, more defensible
  conclusion: **check per-class recall explicitly every time**, don't
  assume class weighting alone fixes it.

## Limitations

Stated directly, not glossed over:

- **Two prevalence rates are still unverified placeholders** — how
  often a caller actually presents with an IMCI-style danger sign has
  no published Nigerian figure. A clinician/CHW estimate would
  replace this with something real.
- **Ward-to-transport data is illustrative**, not sourced from a real
  LGA health office.
- **SQLite session logging is not persistent** on Render's free tier
  — it resets on every restart/redeploy. Production would use a
  managed database.
- **Sandbox SMS never reaches a real phone** — Africa's Talking's
  sandbox mode logs sends but does not deliver to real numbers by
  design; this is a platform constraint, not a bug.
- **USSD sandbox is simulator-only** — dialing a real short code from
  a real phone requires a paid, MNO-approved shortcode, which is a
  production step outside hackathon scope.
- This validates internal logical consistency against the rules it
  was trained to imitate — **it has not been checked against any real
  patient outcome.**

## Roadmap

- [ ] Replace remaining placeholder prevalence rates with clinician
      input
- [ ] Real ward-to-transport data for an actual target LGA
- [ ] Persistent database for session logging
- [ ] Apply for a production shortcode once validated further

## Context

Built for the World Bank / Hack-Nation Small AI for Development
Hackathon Challenge, Health track — part of the Global AI & Digital
Summit 2026.

**Author:** Micheal Adeniyi
