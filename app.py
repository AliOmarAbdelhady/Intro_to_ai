import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mental Productivity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/best_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    feature_df = pd.read_csv("outputs/feature_columns.csv")
    feature_names = feature_df['feature'].tolist()
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# -----------------------------
# UI HEADER
# -----------------------------
st.markdown("""
<div style='text-align:center; padding: 15px;'>
    <h1 style='color:white;'>🧠 Mental Productivity Predictor</h1>
    <h4 style='color:gray;'>AI-powered lifestyle productivity estimation</h4>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# LAYOUT: Left (Inputs) | Right (Summary)
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("📥 Enter Your Daily Lifestyle Metrics")

    inputs = {}

    for feature in feature_names:
        # Skip engineered features – will compute them later
        if feature in [
            "sleep_exercise", "stress_screen_ratio", "diet_mood_product",
            "sleep_squared", "health_score"
        ]:
            continue

        label = feature.replace("_", " ").title()

        if "hours" in feature:
            inputs[feature] = st.number_input(label, 0.0, 24.0, 7.0)
        elif "mins" in feature:
            inputs[feature] = st.number_input(label, 0, 300, 30)
        elif "1_10" in feature:
            inputs[feature] = st.slider(label, 1, 10, 5)
        else:
            inputs[feature] = st.number_input(label, 0.0, 10000.0, 0.0)

# -----------------------------
# Compute Engineered Features
# -----------------------------
sleep = inputs["sleep_hours"]
exercise = inputs["daily_exercise_mins"]
screen = inputs["screen_time_hours"]
stress = inputs["stress_level_1_10"]
mood = inputs["mood_level_1_10"]
diet = inputs["diet_quality_1_10"]

engineered = {
    "sleep_exercise": sleep * exercise,
    "stress_screen_ratio": stress / (screen + 0.001),
    "diet_mood_product": diet * mood,
    "sleep_squared": sleep ** 2,
    "health_score": sleep + diet - stress,
}

all_features = {**inputs, **engineered}
X_input = pd.DataFrame([all_features])[feature_names]
X_scaled = scaler.transform(X_input)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
with right:
    st.subheader("📊 Prediction Summary")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Productivity", use_container_width=True):
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]

        label = "High Productivity" if pred == 1 else "Low Productivity"
        confidence = max(prob)

        # -----------------------------
        # Result Card
        # -----------------------------
        card_color = "#002412" if pred == 1 else "#1E0301"
        emoji = "🟢" if pred == 1 else "🔴"

        st.markdown(
            f"""
            <div style="
                background-color:{card_color};
                padding:20px;
                border-radius:12px;
                text-align:center;
                border: 1px solid #ccc;">
                <h2>{emoji} Prediction: <b>{label}</b></h2>
                <h4>Confidence: {confidence*100:.2f}%</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Inputs Summary Card
        # -----------------------------
        st.markdown("<br><b>📌 Your Inputs:</b>", unsafe_allow_html=True)

        summary_df = pd.DataFrame(all_features.items(), columns=["Feature", "Value"])
        st.dataframe(summary_df, use_container_width=True)


