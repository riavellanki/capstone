import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from load_encrypted_model import load_encrypted_model

def heart_ui():
    # -----------------------
    # Load model and metadata
    # -----------------------
    try:
        pipeline = load_encrypted_model(
            "models/heart/heart_encrypted.bin",
            "heart_key"
        )  # Use path to your saved pipeline
        with open("models/heart/heart_meta.json", "r") as f:
            meta = json.load(f)
        model = pipeline["model"]
        scaler = pipeline["scaler"]
        le_dict = pipeline["label_encoders"]
        features = pipeline["features"]
    except Exception as e:
        st.error(f"Could not load model artifacts. {e}")
        st.stop()

    # -----------------------
    # App setup
    # -----------------------
    #st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
    st.title("❤️ Heart Disease Risk Predictor")
    st.caption("For educational purposes only — not medical advice.")

    st.subheader("Enter Health Information")

    # -----------------------
    # User inputs
    # -----------------------
    age = st.number_input("Age", min_value=1, max_value=120, value=60)
    sex = st.selectbox("Sex", list(le_dict['Sex'].classes_))
    chest_pain = st.selectbox("Chest Pain Type", list(le_dict['ChestPainType'].classes_))
    resting_bp = st.number_input("Resting BP (mmHg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=600, value=200)
    fasting_bs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=500, value=100)
    resting_ecg = st.selectbox("Resting ECG", list(le_dict['RestingECG'].classes_))
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", list(le_dict['ExerciseAngina'].classes_))
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", list(le_dict['ST_Slope'].classes_))


    # -----------------------
    # Prediction button
    # -----------------------
    if st.button("Predict Risk"):
        try:
            # Prepare input DataFrame
            input_dict = {
                "Age": age,
                "Sex": le_dict['Sex'].transform([sex])[0],
                "ChestPainType": le_dict['ChestPainType'].transform([chest_pain])[0],
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": le_dict['RestingECG'].transform([resting_ecg])[0],
                "MaxHR": max_hr,
                "ExerciseAngina": le_dict['ExerciseAngina'].transform([exercise_angina])[0],
                "Oldpeak": oldpeak,
                "ST_Slope": le_dict['ST_Slope'].transform([st_slope])[0],
            }

            input_df = pd.DataFrame([input_dict])

            # Scale numeric features
            numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            # Predict probability
            prob = model.predict_proba(input_df)[0, 1]
            threshold = meta.get("threshold", 0.5)
            # Determine risk level
            if prob < 0.33:
                risk_level = "Low"
                color = "🟩"
            elif prob < 0.66:
                risk_level = "Moderate"
                color = "🟨"
            else:
                risk_level = "High"
                color = "🟥"
            # Display results
            st.markdown(f"### {color} Risk Level: **{risk_level} ({prob*100:.2f}%)**")
            with st.expander("What this means"):
                st.write(
                    "- This tool estimates the risk of heart disease from your inputs.\n"
                    "- It does **not** provide a diagnosis.\n"
                    "- Discuss results with a healthcare professional."
                )
            with st.expander("Input Summary"):
                st.json(input_dict)

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")


if __name__ == "__main__":
    # When run directly with `streamlit run heart_disease_app.py`, call the UI
    heart_ui()