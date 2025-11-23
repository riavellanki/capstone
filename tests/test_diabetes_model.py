# tests/test_diabetes_model.py

import os
import sys
import numpy as np
import pandas as pd
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import diabetes_model as dm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


#Fixture

@pytest.fixture
def dummy_diabetes_df():
    """
    Small synthetic dataset with the same columns your pipeline expects.
    This avoids depending on the real CSV during unit tests.
    """
    n = 40  # keep it small but big enough for stratify

    data = {
        "age": np.linspace(25, 65, n),
        "bmi": np.linspace(20, 35, n),
        "HbA1c_level": np.linspace(5.0, 9.0, n),
        "blood_glucose_level": np.linspace(90, 200, n),
        "gender": ["Male", "Female"] * (n // 2),
        "smoking_history": ["never", "current", "former", "No Info"] * (n // 4),
        "hypertension": [0, 1] * (n // 2),
        "heart_disease": [0, 1] * (n // 2),
        "diabetes": [0, 1] * (n // 2),  # balanced labels for stratify
    }

    df = pd.DataFrame(data)
    return df


# ---------- Tests for core functions ----------

def test_clean_data_removes_duplicates(dummy_diabetes_df):
    df = pd.concat([dummy_diabetes_df, dummy_diabetes_df], ignore_index=True)
    assert len(df) == 80

    cleaned = dm.clean_data(df)
    assert len(cleaned) == 40  # duplicates removed
    # make sure original df not modified in-place
    assert len(df) == 80


def test_preprocess_data_shapes_and_columns(dummy_diabetes_df):
    X, y, features, le_gender, le_smoke = dm.preprocess_data(dummy_diabetes_df)

    # Check that X has expected columns
    expected_features = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender",
        "smoking_history",
        "hypertension",
        "heart_disease",
    ]
    assert features == expected_features
    assert list(X.columns) == expected_features

    # y should be the diabetes target
    assert y.name == "diabetes"
    assert len(X) == len(y)

    # Encoders should have more than one class
    assert len(le_gender.classes_) >= 1
    assert len(le_smoke.classes_) >= 1

    # gender & smoking_history should now be numeric
    assert np.issubdtype(X["gender"].dtype, np.number)
    assert np.issubdtype(X["smoking_history"].dtype, np.number)


def test_split_data_splits_correctly(dummy_diabetes_df):
    X, y, *_ = dm.preprocess_data(dummy_diabetes_df)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(X, y)

    # Total rows should be preserved
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)

    # Shapes of X and y should match within each split
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_scale_data_shapes(dummy_diabetes_df):
    X, y, *_ = dm.preprocess_data(dummy_diabetes_df)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(X, y)

    X_train_s, X_val_s, X_test_s, scaler = dm.scale_data(X_train, X_val, X_test)

    # Same number of rows
    assert X_train_s.shape[0] == X_train.shape[0]
    assert X_val_s.shape[0] == X_val.shape[0]
    assert X_test_s.shape[0] == X_test.shape[0]

    # Same number of features
    assert X_train_s.shape[1] == X_train.shape[1]
    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_")


def test_train_model_returns_logistic_regression(dummy_diabetes_df):
    X, y, *_ = dm.preprocess_data(dummy_diabetes_df)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(X, y)
    X_train_s, X_val_s, X_test_s, scaler = dm.scale_data(X_train, X_val, X_test)

    model = dm.train_model(X_train_s, y_train)
    assert isinstance(model, LogisticRegression)
    # Should be able to predict on training data without error
    preds = model.predict(X_train_s)
    assert len(preds) == len(y_train)


def test_evaluate_model_runs_without_error(dummy_diabetes_df, capsys):
    """
    We don't assert exact metric values, just that the function
    runs and prints metrics without raising exceptions.
    """
    X, y, *_ = dm.preprocess_data(dummy_diabetes_df)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(X, y)
    X_train_s, X_val_s, X_test_s, scaler = dm.scale_data(X_train, X_val, X_test)

    model = dm.train_model(X_train_s, y_train)
    threshold = 0.5

    dm.evaluate_model(model, X_test_s, y_test, threshold)

    captured = capsys.readouterr()
    # Quick sanity check that some metrics were printed
    assert "Accuracy" in captured.out
    assert "Classification report" in captured.out


def test_save_artifacts_creates_files(tmp_path, dummy_diabetes_df, monkeypatch):
    """
    Test that save_artifacts() actually writes the pipeline and meta files.
    We redirect file paths into a temporary directory so we don't touch
    your real project files.
    """
    # Prepare a small trained model & scaler
    X, y, features, *_ = dm.preprocess_data(dummy_diabetes_df)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(X, y)
    X_train_s, X_val_s, X_test_s, scaler = dm.scale_data(X_train, X_val, X_test)
    model = dm.train_model(X_train_s, y_train)
    threshold = 0.7

    # Change working directory temporarily to tmp_path
    # so that diabetes_rf_pipeline.joblib & diabetes_meta.json
    # are written there instead of your real project root.
    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        dm.save_artifacts(model, scaler, features, threshold)

        pipeline_path = tmp_path / "diabetes_logreg_pipeline.joblib"
        meta_path = tmp_path / "diabetes_meta.json"

        assert pipeline_path.exists(), "Pipeline file was not created"
        assert meta_path.exists(), "Meta JSON file was not created"

        # Verify contents of the saved pipeline
        import joblib, json

        pipeline = joblib.load(pipeline_path)
        assert isinstance(pipeline, dict)
        assert "model" in pipeline and "scaler" in pipeline and "features" in pipeline

        meta = json.loads(meta_path.read_text())
        assert "features" in meta and "threshold" in meta
    finally:
        os.chdir(old_cwd)


def test_full_pipeline_end_to_end(dummy_diabetes_df):
    """
    End-to-end smoke test: preprocess -> split -> scale -> train -> predict_proba.
    Just checks that nothing crashes and probabilities are between 0 and 1.
    """
    X, y, features, *_ = dm.preprocess_data(dummy_diabetes_df)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(X, y)
    X_train_s, X_val_s, X_test_s, scaler = dm.scale_data(X_train, X_val, X_test)
    model = dm.train_model(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    assert probs.shape[0] == X_test_s.shape[0]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
