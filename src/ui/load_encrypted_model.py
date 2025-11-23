import streamlit as st
from cryptography.fernet import Fernet
import pickle
import joblib
import io

def load_encrypted_model(path: str, secret_key: str):
    key = st.secrets[secret_key]
    cipher = Fernet(key)

    with open(path, "rb") as f:
        encrypted = f.read()

    decrypted = cipher.decrypt(encrypted)

    # joblib.load() requires a file-like object → wrap bytes in BytesIO
    buffer = io.BytesIO(decrypted)
    model = joblib.load(buffer)

    return model
