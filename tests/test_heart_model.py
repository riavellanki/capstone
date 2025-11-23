import os
import sys
import numpy as np
import pandas as pd
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import heart_model as hm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Fixture: synthetic heart dataset
# ---------------------------
@pytest.fixture
def dummy_heart_df():
    """
    Creates a small synthetic dataset with the same columns
    the heart model expects. Ensures stable and deterministic
    behavior while avoiding dependence on a real CSV.
    """
    n = 40
    data = {
        "Age": np.linspace(29, 77, n),
        "Sex": ["Male", "Female"] * (n // 2),  # length 40
        "ChestPainType": ["typical", "atypical", "non-anginal", "asymptomatic"] * (n // 4),  # length 40
        "RestingBP": np.linspace(110, 180, n),
        "Cholesterol": np.linspace(180, 300, n),
        "FastingBS": [0, 1] * (n // 2),
        "RestingECG": ["normal", "ST", "LVH", "other"] * (n // 4),
        "MaxHR": np.linspace(100, 200, n),
        "ExerciseAngina": ["Y", "N"] * (n // 2),
        "Oldpeak": np.linspace(0.0, 5.0, n),
        "ST_Slope": ["Up", "Flat", "Down", "Up"] * (n // 4),
        "HeartDisease": [0, 1] * (n // 2)
    }

    df = pd.DataFrame(data)
    return df



# ---------------------------
# Tests for core functions
# ---------------------------

def test_preprocess_data_outputs_correct_format(dummy_heart_df):
    X, y, numeric_cols, le_dict = hm.preprocess_data(dummy_heart_df)

    # Ensure target is intact
    assert y.name == "HeartDisease"
    assert len(X) == len(y)

    # All label encoders created
    assert isinstance(le_dict, dict)
    assert len(le_dict) > 0

    # Encoded categorical cols must be numeric
    # Only check columns that are actually in X (i.e., features)
    for col in le_dict.keys():
        if col in X.columns:  # skip target
            assert np.issubdtype(X[col].dtype, np.number)

    # numeric_cols should be a subset of X columns
    for col in numeric_cols:
        assert col in X.columns

    # Ensure target is dropped from features
    assert "HeartDisease" not in X.columns


def test_split_data_splits_correctly(dummy_heart_df):
    X, y, *_ = hm.preprocess_data(dummy_heart_df)
    X_train, X_test, y_train, y_test = hm.split_data(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_scale_data_shapes(dummy_heart_df):
    X, y, numeric_cols, *_ = hm.preprocess_data(dummy_heart_df)
    X_train, X_test, y_train, y_test = hm.split_data(X, y)

    X_train_s, X_test_s, scaler = hm.scale_data(X_train, X_test, numeric_cols)

    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape
    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_")


def test_train_model_returns_random_forest(dummy_heart_df):
    X, y, numeric_cols, *_ = hm.preprocess_data(dummy_heart_df)
    X_train, X_test, y_train, y_test = hm.split_data(X, y)
    X_train_s, X_test_s, scaler = hm.scale_data(X_train, X_test, numeric_cols)

    model = hm.train_model(X_train_s, y_train)

    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X_train_s)
    assert len(preds) == len(y_train)


def test_evaluate_model_runs_without_error(dummy_heart_df, capsys):
    X, y, numeric_cols, *_ = hm.preprocess_data(dummy_heart_df)
    X_train, X_test, y_train, y_test = hm.split_data(X, y)
    X_train_s, X_test_s, scaler = hm.scale_data(X_train, X_test, numeric_cols)

    model = hm.train_model(X_train_s, y_train)

    hm.evaluate_model(model, X_test_s, y_test, threshold=0.5)

    captured = capsys.readouterr()
    assert "Accuracy" in captured.out
    assert "ROC-AUC" in captured.out
    assert "Classification Report" in captured.out


def test_save_artifacts_creates_files(tmp_path, dummy_heart_df):
    X, y, numeric_cols, le_dict = hm.preprocess_data(dummy_heart_df)
    X_train, X_test, y_train, y_test = hm.split_data(X, y)
    X_train_s, X_test_s, scaler = hm.scale_data(X_train, X_test, numeric_cols)
    model = hm.train_model(X_train_s, y_train)

    # Temporarily switch to test directory
    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        hm.save_artifacts(model, scaler, le_dict, list(X.columns), threshold=0.5)

        assert (tmp_path / "heart_pipeline.joblib").exists()
        assert (tmp_path / "heart_meta.json").exists()
    finally:
        os.chdir(old_cwd)


# ---------------------------
# Full End-to-End Test
# ---------------------------

def test_full_pipeline_end_to_end(dummy_heart_df):
    X, y, numeric_cols, le_dict = hm.preprocess_data(dummy_heart_df)
    X_train, X_test, y_train, y_test = hm.split_data(X, y)
    X_train_s, X_test_s, scaler = hm.scale_data(X_train, X_test, numeric_cols)
    model = hm.train_model(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]

    assert probs.shape[0] == X_test_s.shape[0]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
