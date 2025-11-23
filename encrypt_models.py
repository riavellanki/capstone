# encrypt_models.py
from cryptography.fernet import Fernet

def encrypt_file(input_path, output_path):
    """Encrypts a file and writes encrypted bytes to output_path."""
    key = Fernet.generate_key()
    cipher = Fernet(key)

    with open(input_path, "rb") as f:
        encrypted = cipher.encrypt(f.read())

    with open(output_path, "wb") as f:
        f.write(encrypted)

    return key.decode()

def main():
    print("\n--- Encrypting diabetes model ---")
    diabetes_key = encrypt_file(
        "models/diabetes/diabetes_logreg_pipeline.joblib",
        "models/diabetes/diabetes_encrypted.bin"
    )
    print("Diabetes model key:", diabetes_key)

    print("\n--- Encrypting heart model ---")
    heart_key = encrypt_file(
        "models/heart/heart_pipeline.joblib",
        "models/heart/heart_encrypted.bin"
    )
    print("Heart model key:", heart_key)

    print("\n💡 Copy BOTH keys into .streamlit/secrets.toml")
    print("⚠️ Then delete or move the original .joblib files!")
    print("--- Encryption complete ---")

if __name__ == "__main__":
    main()
