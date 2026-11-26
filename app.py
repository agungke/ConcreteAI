from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "cement_kg_m3", "slag_kg_m3", "fly_ash_kg_m3", "water_kg_m3",
    "superplasticizer_kg_m3", "coarse_aggregate_kg_m3",
    "fine_aggregate_kg_m3", "age_days",
]

st.set_page_config(page_title="ConcreteAI", page_icon="🏗️", layout="wide")
st.title("🏗️ ConcreteAI")
st.caption("Concrete Compressive Strength Prediction")
st.write(
    "Estimasi kuat tekan beton berdasarkan komposisi material dan umur curing. "
    "Pilih model, masukkan komposisi campuran, lalu jalankan prediksi."
)

choice = st.sidebar.selectbox("Model prediction", ["Random Forest", "Linear Regression"])
model_file = ROOT / "models" / (
    "random_forest.joblib" if choice == "Random Forest" else "linear_regression.joblib"
)
if not model_file.exists():
    st.error("Model belum tersedia. Jalankan workflow training atau python src/train.py.")
    st.stop()

model = joblib.load(model_file)

# Bounds reflect the observed min/max values in the UCI training dataset.
inputs = [
    ("Cement", "cement_kg_m3", 350.0, 102.0, 540.0, "kg/m³"),
    ("Blast Furnace Slag", "slag_kg_m3", 0.0, 0.0, 359.4, "kg/m³"),
    ("Fly Ash", "fly_ash_kg_m3", 0.0, 0.0, 200.1, "kg/m³"),
    ("Water", "water_kg_m3", 190.0, 121.8, 247.0, "kg/m³"),
    ("Superplasticizer", "superplasticizer_kg_m3", 5.0, 0.0, 32.2, "kg/m³"),
    ("Coarse Aggregate", "coarse_aggregate_kg_m3", 1000.0, 801.0, 1145.0, "kg/m³"),
    ("Fine Aggregate", "fine_aggregate_kg_m3", 750.0, 594.0, 992.6, "kg/m³"),
    ("Concrete Age", "age_days", 28.0, 1.0, 365.0, "days"),
]

st.subheader("Concrete mix composition")
st.caption("Rentang input dibatasi pada rentang observasi dataset UCI; ini tidak menjamin kombinasi campuran layak secara teknis.")
columns = st.columns(4)
values = {}
for i, (label, key, default, minimum, maximum, unit) in enumerate(inputs):
    with columns[i % 4]:
        values[key] = st.number_input(
            f"{label} ({unit})",
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
            step=1.0 if key == "age_days" else 0.1,
            key=key,
        )

if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
    row = pd.DataFrame([[values[key] for key in FEATURES]], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    st.metric("Predicted Concrete Compressive Strength", f"{prediction:.2f} MPa")
    st.info(f"Model used: {choice}")

metrics_path = ROOT / "results" / "metrics.json"
if metrics_path.exists():
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Model evaluation — held-out test set")
    selected = results["metrics"][choice]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f'{selected["test"]["MAE"]:.2f} MPa')
    m2.metric("RMSE", f'{selected["test"]["RMSE"]:.2f} MPa')
    m3.metric("R²", f'{selected["test"]["R2"]:.3f}')

    comparison = pd.DataFrame(
        [{"Model": name, **values["test"]} for name, values in results["metrics"].items()]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.subheader("Evaluation visualizations")
    left, right = st.columns(2)
    diagnostic = ROOT / "results" / "prediction_diagnostics.png"
    importance = ROOT / "results" / "feature_importance.png"
    if diagnostic.exists():
        left.image(str(diagnostic), caption="Actual vs predicted and residual diagnostics", use_container_width=True)
    if importance.exists():
        right.image(str(importance), caption="Random Forest feature importance", use_container_width=True)

st.warning(
    "Academic prototype only. Predictions are estimates from historical data and must not replace "
    "laboratory compressive-strength testing, applicable standards, structural design or engineering judgement."
)
