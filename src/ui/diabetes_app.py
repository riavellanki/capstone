import streamlit as st
import joblib
import json
import pandas as pd
import os
import requests
import joblib
from load_encrypted_model import load_encrypted_model

def download_model(url, local_path):

    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        r = requests.get(url)
        r.raise_for_status()  # fail fast if download fails
        with open(local_path, "wb") as f:
            f.write(r.content)

def diabetes_ui():
    # Load model and metadata
    try:
        #pipeline = joblib.load("models/diabetes/diabetes_rf_pipeline.joblib")
        #with open("models/diabetes/diabetes_meta.json", "r") as f:
        pipeline = load_encrypted_model(
            "models/diabetes/diabetes_encrypted.bin",
            "diabetes_key"
        )
        with open("models/diabetes/diabetes_meta.json", "r") as f:
            meta = json.load(f)
        model = pipeline["model"]
        scaler = pipeline["scaler"]
        features = pipeline["features"]
    except Exception as e:
        st.error(f"Could not load model artifacts. {e}")
        st.stop()

    # App setup
    #st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="💉", layout="centered")
    st.title("💉 Diabetes Risk Predictor")
    st.caption("For educational use only — not medical advice.")

    st.subheader("Enter Health Information")

    # User inputs
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.3, step=0.1)
    hba1c = st.number_input("HbA1c (%)", min_value=3.5, max_value=20.0, value=6.2, step=0.1)
    glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50.0, max_value=500.0, value=145.0, step=1.0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    smoking = st.selectbox("Smoking history", ["never", "No Info", "current", "former", "ever", "not current"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart disease", ["No", "Yes"])

    if st.button("Predict"):
        try:
            # Encode categorical variables
            gender_map = {"Male": 1, "Female": 0, "Other": 2}
            smoking_map = {"never": 0, "No Info": 1, "current": 2, "former": 3, "ever": 4, "not current": 5}
            hyper_map = {"No": 0, "Yes": 1}
            heart_map = {"No": 0, "Yes": 1}

            # Use exact feature names from metadata
            cols = meta["features"]

            # Create dataframe for scaler (fixes warning)
            input_data = pd.DataFrame([{
                "age": int(age),
                "bmi": float(bmi),
                "HbA1c_level": float(hba1c),
                "blood_glucose_level": float(glucose),
                "gender": int(gender_map[gender]),
                "smoking_history": int(smoking_map[smoking]),
                "hypertension": int(hyper_map[hypertension]),
                "heart_disease": int(heart_map[heart_disease])
            }])[cols]

            input_data["HbA1c_level"] = input_data["HbA1c_level"] * 1.025
            scaled = scaler.transform(input_data)
            prob = model.predict_proba(scaled)[0][1]

            if float(hba1c) > 6.0:
                prob = max(prob, 0.20)

            threshold = meta.get("threshold", 0.5)
            #threshold = 0.1
            if prob < 0.30:
                risk_level = "Low"
                color = "🟩"
            elif prob < 0.55:
                risk_level = "Moderate"
                color = "🟨"
            else:
                risk_level = "High"
                color = "🟥"
            st.markdown(f"### {color} Risk Level: **{risk_level} ({prob*100:.2f}%)**")

            with st.expander("What this means"):
                st.write(
                    "- This tool estimates diabetes risk from your inputs.\n"
                    "- It does **not** provide a diagnosis.\n"
                    "- Discuss results with a healthcare professional."
                )

            with st.expander("Input snapshot"):
                st.json({
                    "Age": age,
                    "BMI": bmi,
                    "HbA1c": hba1c,
                    "Blood Glucose": glucose,
                    "Gender": gender,
                    "Smoking": smoking,
                    "Hypertension": hypertension,
                    "Heart Disease": heart_disease
                })

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

if __name__ == "__main__":
    diabetes_ui()