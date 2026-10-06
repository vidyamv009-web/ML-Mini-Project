"""
Cardiovascular Disease Prediction — Flask web app
Loads the pipeline saved from the Jupyter notebook and serves a prediction form.
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent


def load_artifact(filename):
    """Look in the webapp folder first, then the MiniProject folder."""
    for folder in (BASE_DIR, PARENT_DIR):
        path = folder / filename
        if path.exists():
            return joblib.load(path), str(path)
    raise FileNotFoundError(
        f"Could not find {filename}. Place it in {BASE_DIR} or {PARENT_DIR}."
    )


model, model_path = load_artifact("cvd_prediction_model.joblib")
feature_cols, cols_path = load_artifact("cvd_feature_columns.joblib")
print("Model loaded from:", model_path)
print("Feature columns:", feature_cols)


def build_feature_row(form):
    """Turn form values into one row with the exact columns the model expects."""
    age = int(form["age"])
    gender = int(form["gender"])  # 1 = female, 2 = male (dataset coding)
    height = float(form["height"])
    weight = float(form["weight"])
    ap_hi = int(form["ap_hi"])
    ap_lo = int(form["ap_lo"])
    cholesterol = int(form["cholesterol"])
    gluc = int(form["gluc"])
    smoke = int(form["smoke"])
    alco = int(form["alco"])
    active = int(form["active"])

    if height <= 0 or weight <= 0:
        raise ValueError("Height and weight must be greater than 0.")
    if ap_hi <= ap_lo:
        raise ValueError("Systolic BP (ap_hi) must be greater than diastolic BP (ap_lo).")

    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo

    raw = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": round(bmi, 4),
        "pulse_pressure": pulse_pressure,
    }

    # Keep only columns the saved model was trained on, in the same order
    row = {col: raw[col] for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols), bmi


def risk_band(probability):
    if probability < 0.35:
        return "Lower risk", "ok"
    if probability < 0.65:
        return "Moderate risk", "mid"
    return "Higher risk", "high"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features, bmi = build_feature_row(request.form)
        pred = int(model.predict(features)[0])
        proba = float(model.predict_proba(features)[0][1])
        band, tone = risk_band(proba)

        if pred == 1:
            headline = "Higher chance of cardiovascular disease"
        else:
            headline = "Lower chance of cardiovascular disease"

        result = {
            "headline": headline,
            "probability": round(proba * 100, 1),
            "band": band,
            "tone": tone,
            "bmi": round(bmi, 1),
            "prediction": pred,
        }
        return render_template("index.html", result=result, form=request.form)
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
            form=request.form,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
