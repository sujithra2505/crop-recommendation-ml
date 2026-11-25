"""
Phase 3 - Data preprocessing for the Crop Recommendation project.

What this module does (in order):
1. Load the raw dataset (read-only - the original CSV is never changed).
2. Separate the 7 input features (X) from the target crop name (y).
3. Encode crop names into numbers with LabelEncoder.
4. Split into 80% train / 20% test, stratified by crop, random_state=42.
5. Scale features with StandardScaler, fitted on the TRAINING data only.
6. Save the encoder, scaler, split data (scaled + unscaled) and a summary.

Other modules (train.py, evaluate.py, predict.py, app.py) import the
constants and helpers from here, so every part of the project uses
exactly the same feature order and preprocessing.

Run from the project root:
    python -m src.data_preprocessing
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths and settings (shared by the whole project)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# The order of features matters: the scaler and models expect this exact order.
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1-2: load data and separate features / target
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and check that the expected columns exist."""
    df = pd.read_csv(path)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")
    return df


def split_features_target(df: pd.DataFrame):
    """Return X (the 7 input features, fixed order) and y (crop names)."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# ---------------------------------------------------------------------------
# Step 3: encode labels
# ---------------------------------------------------------------------------
def encode_labels(y: pd.Series):
    """
    Convert crop names to integers (apple -> 0, banana -> 1, ...).
    LabelEncoder sorts names alphabetically. It only learns the list of
    crop names, not anything about the feature values, so fitting it on all
    labels does not leak information from the test set.
    """
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), index=y.index, name=TARGET)
    return y_encoded, encoder


# ---------------------------------------------------------------------------
# Step 4: stratified train/test split
# ---------------------------------------------------------------------------
def make_train_test_split(X, y):
    """
    80/20 split. stratify=y keeps every crop at the same share (80 train /
    20 test per crop) so both sets stay balanced, as found in the EDA.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ---------------------------------------------------------------------------
# Step 5: scaling (fit on train only -> no data leakage)
# ---------------------------------------------------------------------------
def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Learn the mean and standard deviation of each feature from the
    TRAINING set only. If we used the test set here, the model would
    indirectly "see" test data before evaluation (data leakage).
    """
    scaler = StandardScaler()
    scaler.fit(X_train)  # fitting on a DataFrame also stores the feature names
    return scaler


def scale_features(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted scaler and keep column names and row index."""
    return pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES, index=X.index)


# ---------------------------------------------------------------------------
# Loading helpers used by later phases
# ---------------------------------------------------------------------------
def load_preprocessors():
    """Load the saved LabelEncoder and StandardScaler."""
    return joblib.load(ENCODER_PATH), joblib.load(SCALER_PATH)


def load_processed_data() -> dict:
    """
    Load the saved split data. Returns a dict with:
    X_train, X_test (unscaled), X_train_scaled, X_test_scaled, y_train, y_test.
    """
    data = {}
    for name in ["X_train", "X_test", "X_train_scaled", "X_test_scaled"]:
        data[name] = pd.read_csv(PROCESSED_DIR / f"{name}.csv", index_col=0)
    for name in ["y_train", "y_test"]:
        data[name] = pd.read_csv(PROCESSED_DIR / f"{name}.csv", index_col=0)[TARGET]
    return data


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_preprocessing(save: bool = True) -> dict:
    """Run every preprocessing step and (optionally) save all outputs."""
    df = load_data()
    X, y = split_features_target(df)
    y_encoded, encoder = encode_labels(y)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y_encoded)

    scaler = fit_scaler(X_train)
    X_train_scaled = scale_features(scaler, X_train)
    X_test_scaled = scale_features(scaler, X_test)

    data = {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
    }

    if save:
        for d in (PROCESSED_DIR, MODELS_DIR, METRICS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        # The original row index is kept so any row can be traced back to the raw CSV.
        for name, frame in data.items():
            frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=True, index_label="row_id")
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)

        summary = build_summary(df, encoder, scaler, data)
        with open(METRICS_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    data["encoder"] = encoder
    data["scaler"] = scaler
    return data


def build_summary(df, encoder, scaler, data) -> dict:
    """Small, human-readable record of what preprocessing produced."""
    train_counts = data["y_train"].value_counts()
    test_counts = data["y_test"].value_counts()
    return {
        "raw_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(data["X_train"])),
        "test_rows": int(len(data["X_test"])),
        "n_classes": int(len(encoder.classes_)),
        "label_mapping": {crop: int(i) for i, crop in enumerate(encoder.classes_)},
        "train_samples_per_class": {"min": int(train_counts.min()), "max": int(train_counts.max())},
        "test_samples_per_class": {"min": int(test_counts.min()), "max": int(test_counts.max())},
        "scaler_fitted_on": "training set only",
        "scaler_mean": dict(zip(FEATURES, scaler.mean_.round(4).tolist())),
        "scaler_std": dict(zip(FEATURES, scaler.scale_.round(4).tolist())),
    }


if __name__ == "__main__":
    result = run_preprocessing(save=True)
    print("Preprocessing complete")
    print(f"  Train: {result['X_train'].shape}, Test: {result['X_test'].shape}")
    print(f"  Classes: {len(result['encoder'].classes_)}")
    print(f"  Saved encoder -> {ENCODER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Saved scaler  -> {SCALER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Saved split data -> {PROCESSED_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  Summary -> {(METRICS_DIR / 'preprocessing_summary.json').relative_to(PROJECT_ROOT)}")
