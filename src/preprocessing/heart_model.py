import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------
# Load Data
# -----------------------
def load_data(filename):
    df = pd.read_csv(filename)
    print("Data loaded. Shape:", df.shape)
    return df


# -----------------------
# Clean Data
# -----------------------
def clean_data(df):
    # Keep the original logic as-is (no cleaning was done originally)
    return df


# -----------------------
# Preprocess Data
# -----------------------
def preprocess_data(df):
    df = df.copy()

    # ORIGINAL CODE – unchanged
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['target']]

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)
    print("Features:", list(X.columns))

    return X, y, numeric_cols, le_dict


# -----------------------
# Split Data
# -----------------------
def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


# -----------------------
# Scale Data
# -----------------------
def scale_data(X_train, X_test, numeric_cols):
    scaler = StandardScaler()

    # ORIGINAL SCALING LOGIC
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler


# -----------------------
# Train Model
# -----------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        min_samples_split=10, min_samples_leaf=4,
        random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("Model trained.")
    return model


# -----------------------
# Evaluate Model
# -----------------------
def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "predictions": y_pred,
        "probabilities": y_prob
    }


# -----------------------
# Save Artifacts
# -----------------------
def save_artifacts(model, scaler, le_dict, features, threshold=0.5):
    pipeline = {
        "model": model,
        "scaler": scaler,
        "label_encoders": le_dict,
        "features": features
    }
    joblib.dump(pipeline, "heart_pipeline.joblib")

    meta = {"features": features, "threshold": threshold}

    with open("heart_meta.json", "w") as f:
        json.dump(meta, f)

    print("Saved heart_pipeline.joblib & heart_meta.json")


# -----------------------
# Main Pipeline
# -----------------------
def run_pipeline(csv_path="../../data/heart.csv", threshold=0.5):
    df = load_data(csv_path)
    df = clean_data(df)
    X, y, numeric_cols, le_dict = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_data(X_train, X_test, numeric_cols)

    model = train_model(X_train_s, y_train)
    results = evaluate_model(model, X_test_s, y_test, threshold)

    save_artifacts(model, scaler, le_dict, list(X.columns), threshold)

    return results


# -----------------------
# Script Execution
# -----------------------
if __name__ == "__main__":
    run_pipeline()
