"""Aplikasi web prediksi risiko stroke (Regresi Logistik).

Jalankan lokal : python app.py
Jalankan produksi : gunicorn app:app
Pastikan folder model/ sudah berisi file hasil notebook:
preprocessor.joblib, model_logreg.joblib, config.json
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Muat preprocessor (imputer + scaler + encoder) dan model hasil training
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "model_logreg.joblib"))
with open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
THRESHOLD = float(config.get("threshold", 0.5))

app = Flask(__name__)

# Pilihan dropdown: (nilai untuk model, label tampilan)
OPTIONS = {
    "gender": [("Male", "Laki-laki"), ("Female", "Perempuan")],
    "hypertension": [("0", "Tidak"), ("1", "Ya")],
    "heart_disease": [("0", "Tidak"), ("1", "Ya")],
    "ever_married": [("Yes", "Pernah menikah"), ("No", "Belum pernah menikah")],
    "work_type": [
        ("Private", "Swasta"),
        ("Self-employed", "Wiraswasta"),
        ("Govt_job", "Pegawai pemerintah"),
        ("children", "Anak-anak"),
        ("Never_worked", "Belum pernah bekerja"),
    ],
    "Residence_type": [("Urban", "Perkotaan"), ("Rural", "Pedesaan")],
    "smoking_status": [
        ("never smoked", "Tidak pernah merokok"),
        ("formerly smoked", "Mantan perokok"),
        ("smokes", "Perokok aktif"),
        ("Unknown", "Tidak diketahui"),
    ],
}

# Batas nilai wajar untuk validasi input angka
LIMITS = {
    "age": (0, 120, "Usia"),
    "avg_glucose_level": (40, 400, "Kadar glukosa rata-rata"),
    "bmi": (10, 100, "BMI"),
}


def validate(form):
    """Ambil dan validasi input form. Return (data, errors)."""
    data, errors = {}, []

    for key, (low, high, label) in LIMITS.items():
        raw = (form.get(key) or "").strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        if not (low <= value <= high):
            errors.append(f"{label} harus antara {low} dan {high}.")
            continue
        data[key] = value

    for key, choices in OPTIONS.items():
        value = form.get(key)
        if value not in [c[0] for c in choices]:
            errors.append(f"Pilihan '{key}' tidak valid.")
            continue
        data[key] = int(value) if key in ("hypertension", "heart_disease") else value

    return data, errors


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors = None, []
    form_values = request.form if request.method == "POST" else {}

    if request.method == "POST":
        data, errors = validate(request.form)
        if not errors:
            row = pd.DataFrame([data])
            X = preprocessor.transform(row)
            proba = float(model.predict_proba(X)[0, 1])
            risky = proba >= THRESHOLD
            result = {
                "label": "Berisiko Stroke" if risky else "Tidak Berisiko",
                "risky": risky,
                "probability": round(proba * 100, 1),
                "threshold": round(THRESHOLD * 100, 1),
            }

    return render_template(
        "index.html",
        options=OPTIONS,
        values=form_values,
        result=result,
        errors=errors,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
