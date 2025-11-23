import streamlit as st
from password_page import password_page
from heart_disease_app import heart_ui
from diabetes_app import diabetes_ui
import time

st.set_page_config(page_title="Multi-disease Risk Predictor", layout="centered")
st.title("🩺 Multi-disease Risk Predictor")
st.markdown("⚠️ This app does NOT store or transmit your health data. All computation is done locally and discarded immediately.")
st.markdown("**Educational tool only — not a medical diagnosis.**")
# First, check authentication
password_page()

# Only show main app if authenticated
if st.session_state.get("authenticated", False):

    # Update last interaction on any user action
    st.session_state.last_interaction = time.time()

    # Select model
    model_choice = st.radio(
        "Select which disease risk to check:",
        ("Heart Disease", "Diabetes"),
        horizontal=True
    )

    if model_choice == "Heart Disease":
        heart_ui()
    else:
        diabetes_ui()
