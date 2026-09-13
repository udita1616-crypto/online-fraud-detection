import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# ============================================================
# 1. PREDEFINED FRAUD-DETECTION RULES
# ============================================================
# These rules are the source of truth for this prototype.
#
# FRAUD:
#   - Amount > ₹150,000 AND night (23:00-05:00) AND international
#     OR
#   - Failed attempts >= 4
#
# SUSPICIOUS:
#   - Failed attempts >= 3
#     OR
#   - Amount > ₹50,000 AND international
#
# SAFE:
#   - Everything else
# ============================================================

def is_night(hour):
    """Return True for 11 PM through 5 AM."""
    return hour >= 23 or hour <= 5


def apply_predefined_rules(amount, hour, is_foreign, failed_attempts):
    """Classify a transaction using only the project's predefined rules."""
    if (
        (amount > 150000 and is_night(hour) and is_foreign == 1)
        or failed_attempts >= 4
    ):
        return 2  # FRAUD

    if (
        failed_attempts >= 3
        or (amount > 50000 and is_foreign == 1)
    ):
        return 1  # SUSPICIOUS

    return 0  # SAFE


# ============================================================
# 2. SELF-GENERATED HISTORICAL DATASET
# ============================================================
# No external dataset is required. The application creates its
# own synthetic historical transaction records at startup.
# Labels are generated from the predefined rules above.
# ============================================================

def generate_historical_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    amount = rng.uniform(500, 200000, n)
    old_balance = amount + rng.uniform(1000, 150000, n)
    hour = rng.integers(0, 24, n)
    is_foreign = rng.choice([0, 1], size=n, p=[0.90, 0.10])
    failed_attempts = rng.choice(
        [0, 1, 2, 3, 4, 5],
        size=n,
        p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.01],
    )
    new_balance = old_balance - amount

    labels = [
        apply_predefined_rules(
            float(a), int(h), int(f), int(fa)
        )
        for a, h, f, fa in zip(
            amount, hour, is_foreign, failed_attempts
        )
    ]

    return pd.DataFrame(
        {
            "amount": amount,
            "old_balance": old_balance,
            "new_balance": new_balance,
            "hour": hour,
            "is_foreign": is_foreign,
            "failed_attempts": failed_attempts,
            "label": labels,
        }
    )


historical_df = generate_historical_dataset()

FEATURES = [
    "amount",
    "old_balance",
    "new_balance",
    "hour",
    "is_foreign",
    "failed_attempts",
]

X = historical_df[FEATURES]
y = historical_df["label"]

# The decision tree is trained only on the generated historical data.
model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X, y)

LABELS = {
    0: {
        "status": "SAFE",
        "class": "safe",
        "message": "The transaction matches the normal behavior pattern.",
    },
    1: {
        "status": "SUSPICIOUS",
        "class": "suspicious",
        "message": "The transaction matches one or more suspicious patterns.",
    },
    2: {
        "status": "FRAUD DETECTED",
        "class": "fraud",
        "message": "The transaction matches a high-risk fraud pattern.",
    },
}


def get_triggered_rules(amount, hour, is_foreign, failed_attempts):
    """Return the exact predefined rules triggered by the transaction."""
    rules = []

    if amount > 150000 and is_night(hour) and is_foreign == 1:
        rules.append(
            "High-value international transaction during night hours "
            "(11 PM - 5 AM)"
        )

    if failed_attempts >= 4:
        rules.append("Failed login attempts are 4 or more")

    if failed_attempts >= 3:
        rules.append("Failed login attempts are 3 or more")

    if amount > 50000 and is_foreign == 1:
        rules.append(
            "International transaction above ₹50,000"
        )

    return rules


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            amount = float(request.form["amount"])
            old_balance = float(request.form["old_balance"])
            hour = int(request.form["hour"])
            failed_attempts = int(request.form["failed_attempts"])
            is_foreign = int(request.form["is_foreign"])

            # Basic input validation.
            if amount <= 0 or old_balance < 0:
                raise ValueError("Amount and balance must be valid positive values.")

            if not 0 <= hour <= 23:
                raise ValueError("Hour must be between 0 and 23.")

            if failed_attempts < 0:
                raise ValueError("Failed attempts cannot be negative.")

            if is_foreign not in (0, 1):
                raise ValueError("Invalid transaction location value.")

            new_balance = max(0.0, old_balance - amount)

            features = pd.DataFrame(
                [[
                    amount,
                    old_balance,
                    new_balance,
                    hour,
                    is_foreign,
                    failed_attempts,
                ]],
                columns=FEATURES,
            )

            # Model prediction from the synthetic historical dataset.
            model_prediction = int(model.predict(features)[0])

            # Rule-based result is authoritative for this prototype.
            rule_prediction = apply_predefined_rules(
                amount,
                hour,
                is_foreign,
                failed_attempts,
            )

            # Keep the explicit predefined rules as the final decision.
            prediction = rule_prediction

            info = LABELS[prediction]
            triggered_rules = get_triggered_rules(
                amount,
                hour,
                is_foreign,
                failed_attempts,
            )

            result = {
                "status": info["status"],
                "class": info["class"],
                "message": info["message"],
                "amount": amount,
                "old_balance": old_balance,
                "new_balance": new_balance,
                "hour": hour,
                "failed_attempts": failed_attempts,
                "is_foreign": "International" if is_foreign else "Domestic",
                "triggered_rules": triggered_rules,
                "model_prediction": LABELS[model_prediction]["status"],
            }

        except (ValueError, KeyError, TypeError) as exc:
            result = {
                "status": "INVALID INPUT",
                "class": "invalid",
                "message": str(exc),
                "triggered_rules": [],
            }

    return render_template(
        "index.html",
        result=result,
        dataset_size=len(historical_df),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
