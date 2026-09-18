"""
Full USSD triage callback for Africa's Talking.

Key idea: USSD is stateless per request. Africa's Talking sends the
full history of what the user has typed so far in `text`, joined by
'*' (e.g. "1*2*1"). We reconstruct the current step by splitting that
string -- there's no session object to hang state on between requests.
"""
import os
import sqlite3
from datetime import datetime
import joblib
import pandas as pd
from flask import Flask, request

app = Flask(__name__)

DB_PATH = "sessions.db"
MODEL_PATH = "triage_tree_final.joblib"
FEATURES_PATH = "feature_columns.csv"

# Load the trained model once at startup. If either file is missing
# (e.g. not yet deployed alongside app.py), the app still runs fine --
# it just skips the model cross-check rather than crashing.
try:
    MODEL = joblib.load(MODEL_PATH)
    FEATURE_COLUMNS = pd.read_csv(FEATURES_PATH, header=None)[0].tolist()
except Exception as e:
    print(f"[MODEL] Could not load trained model, cross-check disabled: {e}")
    MODEL = None
    FEATURE_COLUMNS = None

WARDS = {
    "1": {"name": "Isa-Ope Ward",   "facility": "Isa-Ope PHC",          "ambulance": False,  "contact": "N/A",                         "eta": "9 mins"},
    "2": {"name": "Idofoi Ward",  "facility": "Ibile PHC",         "ambulance": True, "contact": "Tricycle Ambulance Rider - Kabiru",     "eta": "14 mins"},
    "3": {"name": "Oke-Ola Ward", "facility": "Oke-Idagba PHC","ambulance": False, "contact": "Community Rider - Olawale",     "eta": "11 mins"},
}

# Each sequence entry: (prompt_text, symptom_key, tier_if_yes)
SEQUENCES = {
    "child": [
        ("Does the child have ANY danger sign (convulsions, cannot drink/breastfeed, vomiting everything, unusually sleepy)?", "danger", "RED"),
        ("Is the child breathing fast or with difficulty?", "breathing", "YELLOW"),
        ("Diarrhea with sunken eyes or very low energy?", "diarrhea_dehydration", "YELLOW"),
        ("Has the fever lasted more than 3 days?", "fever_gt3d", "YELLOW"),
    ],
    "adult": [
        ("Do you have chest pain with breathlessness, coughing blood, severe dehydration, or sudden confusion?", "danger", "RED"),
        ("Cough or fever lasting more than 2 weeks?", "cough_fever_gt2wk", "YELLOW"),
    ],
    "pregnant": [
        ("Do you have vaginal bleeding, severe headache/blurred vision, convulsions, or reduced fetal movement?", "danger", "RED"),
        ("Swelling of hands/face/feet along with headache?", "swelling_headache", "YELLOW"),
    ],
}

WHO_LABELS = {"1": "child", "2": "adult", "3": "pregnant"}

SYMPTOM_KEYS = ["danger", "breathing", "diarrhea_dehydration",
                "fever_gt3d", "cough_fever_gt2wk", "swelling_headache"]


def build_feature_row(who, true_symptom_key=None):
    """
    Build the one-row feature vector the model expects, matching the
    exact column set/order it was trained on (FEATURE_COLUMNS).
    true_symptom_key: which single symptom was answered "Yes" (if any) --
    matches how the synthetic training data represented each case.
    """
    row = {k: 0 for k in SYMPTOM_KEYS}
    if true_symptom_key:
        row[true_symptom_key] = 1
    row["who_adult"] = 1 if who == "adult" else 0
    row["who_child"] = 1 if who == "child" else 0
    row["who_pregnant"] = 1 if who == "pregnant" else 0
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def model_cross_check(who, true_symptom_key, rule_tier):
    """
    Ask the trained model the same question and log agreement/disagreement.
    Never affects what the caller sees -- the rules stay authoritative.
    """
    if MODEL is None:
        return None
    try:
        features = build_feature_row(who, true_symptom_key)
        model_tier = MODEL.predict(features)[0]
        if model_tier != rule_tier:
            print(f"[MODEL MISMATCH] rules={rule_tier} model={model_tier} "
                  f"who={who} symptom={true_symptom_key}")
        return model_tier
    except Exception as e:
        print(f"[MODEL] prediction failed, continuing without it: {e}")
        return None


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT, ward TEXT, who TEXT, tier TEXT, model_tier TEXT,
            detail TEXT, timestamp TEXT
        )
    """)
    # For a pre-existing local sessions.db from before this column existed
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN model_tier TEXT")
    except sqlite3.OperationalError:
        pass  # column already there
    conn.commit()
    conn.close()


def log_session(phone, ward_name, who, tier, model_tier, detail):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (phone, ward, who, tier, model_tier, detail, timestamp) VALUES (?,?,?,?,?,?,?)",
        (phone, ward_name, who, tier, model_tier, detail, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def send_red_alert(phone, ward):
    """
    Stub for the real SMS-out call. Requires:
      pip install africastalking
      africastalking.initialize(username="sandbox", api_key="YOUR_SANDBOX_API_KEY")
    In sandbox mode, SMS only reaches numbers you've registered under
    'Simulator Numbers' on the dashboard -- it will NOT reach a real
    phone until you have production credentials.
    """
    print(f"[ALERT] Would SMS {ward['facility']} ({ward['contact']}): "
          f"Red case at {phone}, ETA {ward['eta']}")
    try:
        import africastalking
        africastalking.initialize(
            username=os.environ.get("AT_USERNAME", "sandbox"),
            api_key=os.environ.get("AT_API_KEY", "")
        )
        sms = africastalking.SMS
        recipient = os.environ.get("AT_ALERT_RECIPIENT", "+234XXXXXXXXXX")
        sms.send(f"RED alert: caller {phone} needs help. Ward: {ward['name']}", [recipient])
    except Exception as e:
        # A failed SMS send must never take down the USSD response --
        # the caller still needs their triage result either way.
        print(f"[ALERT] SMS send failed (caller still gets their result): {e}")


@app.route("/ussd", methods=["POST"])
def ussd():
    session_id = request.values.get("sessionId", "")
    phone = request.values.get("phoneNumber", "")
    text = request.values.get("text", "")

    parts = text.split("*") if text else []

    # Step 0: no input yet -> ward menu
    if len(parts) == 0:
        return "CON Welcome to Health Check.\nWhich ward are you calling from?\n1: Umuoji Ward\n2: Nkwelle Ward\n3: Ogbunike Ward"

    ward = WARDS.get(parts[0])
    if ward is None:
        return "END Invalid ward selection."

    # Step 1: ward chosen -> who menu
    if len(parts) == 1:
        return "CON Who is this check for?\n1: Child under 5\n2: Adult\n3: Pregnant woman"

    who = WHO_LABELS.get(parts[1])
    if who is None:
        return "END Invalid selection."

    sequence = SEQUENCES[who]
    answers = parts[2:]  # each subsequent input is a yes(1)/no(2) answer

    # Walk the sequence against the answers given so far
    for i, (prompt, symptom_key, tier) in enumerate(sequence):
        if i < len(answers):
            answer = answers[i]
            if answer == "1":  # Yes -> tier determined
                detail = f"{symptom_key} = yes"
                model_tier = model_cross_check(who, symptom_key, tier)
                log_session(phone, ward["name"], who, tier, model_tier, detail)
                if tier == "RED":
                    send_red_alert(phone, ward)
                    transport = (f"Ambulance from {ward['facility']} notified, ETA ~{ward['eta']}."
                                 if ward["ambulance"] else
                                 f"No ambulance on-site. Nearest transport: {ward['contact']}, ETA ~{ward['eta']}.")
                    return f"END URGENT - GO NOW.\n{transport}\nShared with: {ward['facility']}"
                else:
                    return f"END SEE A HEALTH WORKER within 24 hours.\n({symptom_key} flagged)"
            elif answer == "2":  # No -> continue to next question
                continue
            else:
                return "END Invalid answer."
        else:
            # this is the next unanswered question -> ask it
            return f"CON {prompt}\n1: Yes\n2: No"

    # all questions answered "No" -> Green
    model_tier = model_cross_check(who, None, "GREEN")
    log_session(phone, ward["name"], who, "GREEN", model_tier, "no danger signs")
    return "END SELF-CARE AT HOME. No urgent signs found. Rest, fluids, recheck if it worsens."


init_db()

if __name__ == "__main__":
    app.run(port=5000, debug=True)
