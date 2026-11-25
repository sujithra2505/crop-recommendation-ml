"""
Phase 6 - Prediction pipeline for the Crop Recommendation project.

Loads the BEST model chosen in Phase 5 (models/best_model.json), plus the
saved scaler and label encoder, and turns 7 soil/weather values into a
crop recommendation. Nothing is retrained here.

The same steps as training are applied, in the same order:
  1. validate the 7 inputs (missing / non-numeric / impossible values)
  2. put them in the exact training feature order
  3. scale them ONLY if the chosen model was trained on scaled data
  4. predict, then decode the number back to a crop name

Usage from the project root (Windows PowerShell):
    python -m src.predict --N 90 --P 42 --K 43 --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 203

Usage from Python:
    from src.predict import CropPredictor
    result = CropPredictor().predict({"N": 90, "P": 42, ...})
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import ENCODER_PATH, FEATURES, MODELS_DIR, SCALER_PATH

BEST_MODEL_CARD = MODELS_DIR / "best_model.json"

# Physically possible limits. Values outside these are rejected as errors.
# They are deliberately wider than the training data: a value can be
# possible but unusual (that only triggers a warning, see below).
HARD_LIMITS = {
    "N": (0.0, 500.0),             # soil nitrogen ratio, cannot be negative
    "P": (0.0, 500.0),             # soil phosphorus ratio
    "K": (0.0, 500.0),             # soil potassium ratio
    "temperature": (-20.0, 60.0),  # °C
    "humidity": (0.0, 100.0),      # relative humidity is a percentage
    "ph": (0.0, 14.0),             # the pH scale runs from 0 to 14
    "rainfall": (0.0, 5000.0),     # mm
}

FEATURE_INFO = {  # labels and units used by the CLI and the app
    "N": ("Nitrogen (N)", "ratio in soil"),
    "P": ("Phosphorus (P)", "ratio in soil"),
    "K": ("Potassium (K)", "ratio in soil"),
    "temperature": ("Temperature", "°C"),
    "humidity": ("Relative humidity", "%"),
    "ph": ("Soil pH", "0-14"),
    "rainfall": ("Rainfall", "mm"),
}

TOP_K = 3


class InputValidationError(ValueError):
    """Raised when inputs are missing or invalid. `.errors` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class PredictionResult:
    crop: str                                  # recommended crop
    confidence: float | None                   # probability of that crop (None if unsupported)
    top_k: list[tuple[str, float]] = field(default_factory=list)  # [(crop, probability), ...]
    warnings: list[str] = field(default_factory=list)              # e.g. value outside training range
    model_name: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _to_number(name: str, value) -> float:
    """Convert one input to float, or raise ValueError with a clear message."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"{name} is missing")
    if isinstance(value, bool):  # True/False are technically numbers in Python - reject them
        raise ValueError(f"{name} must be a number, got {value!r}")
    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ValueError(f"{name} must be a number, got {value!r}") from None
    else:
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def validate_inputs(raw: dict) -> dict[str, float]:
    """
    Check all 7 inputs and return them as floats in the training order.
    Collects every problem and raises one InputValidationError listing them all,
    so a user can fix everything at once.
    """
    if not isinstance(raw, dict):
        raise InputValidationError(["inputs must be a dictionary of feature name -> value"])

    errors = []
    unknown = sorted(set(raw) - set(FEATURES))
    if unknown:
        errors.append(f"unknown input(s): {', '.join(unknown)} (expected: {', '.join(FEATURES)})")

    clean = {}
    for name in FEATURES:
        try:
            value = _to_number(name, raw.get(name))
        except ValueError as e:
            errors.append(str(e))
            continue
        low, high = HARD_LIMITS[name]
        if not low <= value <= high:
            errors.append(f"{name} = {value:g} is outside the valid range {low:g} to {high:g}")
            continue
        clean[name] = value

    if errors:
        raise InputValidationError(errors)
    return clean


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class CropPredictor:
    """Loads the saved best model + preprocessing objects once, then predicts many times."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        models_dir = Path(models_dir)
        card_path = models_dir / BEST_MODEL_CARD.name
        if not card_path.exists():
            raise FileNotFoundError(
                f"{card_path} not found. Run: python -m src.data_preprocessing, "
                "python -m src.train, python -m src.evaluate")
        self.card = json.loads(card_path.read_text(encoding="utf-8"))
        self.model_name = self.card["model_name"]
        self.uses_scaled = self.card["uses_scaled"]
        self.model = joblib.load(models_dir / self.card["model_file"])
        self.encoder = joblib.load(models_dir / ENCODER_PATH.name)
        self.scaler = joblib.load(models_dir / SCALER_PATH.name)
        self.training_ranges = self.card["training_feature_ranges"]
        self._check_consistency()

    def _check_consistency(self) -> None:
        """Fail early if the saved files do not belong together."""
        if self.card["features"] != FEATURES:
            raise ValueError("Feature list in best_model.json does not match the code")
        if list(getattr(self.model, "feature_names_in_", FEATURES)) != FEATURES:
            raise ValueError("Saved model was trained with a different feature order")
        if list(self.scaler.feature_names_in_) != FEATURES:
            raise ValueError("Saved scaler was fitted with a different feature order")
        if len(self.model.classes_) != len(self.encoder.classes_):
            raise ValueError("Model and label encoder have a different number of crops")

    @property
    def supports_probability(self) -> bool:
        return hasattr(self.model, "predict_proba")

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact same transformation as training: fixed column order, scale only if needed."""
        X = X[FEATURES].astype(float)
        if self.uses_scaled:
            X = pd.DataFrame(self.scaler.transform(X), columns=FEATURES, index=X.index)
        return X

    def range_warnings(self, clean: dict[str, float]) -> list[str]:
        """Warn (not fail) when a value is outside what the model saw during training."""
        out = []
        for name, value in clean.items():
            r = self.training_ranges[name]
            if not r["min"] <= value <= r["max"]:
                out.append(f"{name} = {value:g} is outside the training range "
                           f"({round(r['min'], 2):g} to {round(r['max'], 2):g}); "
                           "the prediction is less reliable.")
        return out

    def predict(self, raw: dict) -> PredictionResult:
        """Validate one set of inputs and return the recommended crop."""
        clean = validate_inputs(raw)
        X = self.prepare_features(pd.DataFrame([clean], columns=FEATURES))
        crop_id = self.model.predict(X)[0]
        crop = str(self.encoder.inverse_transform([crop_id])[0])

        confidence, top_k = None, []
        if self.supports_probability:
            probs = self.model.predict_proba(X)[0]
            names = self.encoder.inverse_transform(self.model.classes_)
            order = np.argsort(probs)[::-1][:TOP_K]
            # Crops with 0 % are not real alternatives, so they are left out.
            top_k = [(str(names[i]), float(probs[i])) for i in order if probs[i] > 0]
            confidence = float(probs[list(names).index(crop)])

        return PredictionResult(crop=crop, confidence=confidence, top_k=top_k,
                                warnings=self.range_warnings(clean), model_name=self.model_name)

    def predict_many(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crop names for a table of already-valid rows (used for testing)."""
        return self.encoder.inverse_transform(self.model.predict(self.prepare_features(X)))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recommend a crop from soil and weather values.")
    for name in FEATURES:
        label, unit = FEATURE_INFO[name]
        # argparse treats "%" as a format code, so it must be written as "%%"
        parser.add_argument(f"--{name}", required=True, help=f"{label} ({unit})".replace("%", "%%"))
    args = vars(parser.parse_args(argv))

    predictor = CropPredictor()
    try:
        result = predictor.predict(args)
    except InputValidationError as e:
        print("Invalid input:")
        for err in e.errors:
            print(f"  - {err}")
        return 1

    print(f"Model: {result.model_name}")
    print(f"Recommended crop: {result.crop}")
    if result.confidence is not None:
        print(f"Model confidence: {result.confidence:.1%} (share of trees voting for it - not a guarantee)")
        print(f"Top {len(result.top_k)} (crops with 0% are not shown):")
        for i, (crop, p) in enumerate(result.top_k, 1):
            print(f"  {i}. {crop:<12} {p:.1%}")
    for w in result.warnings:
        print(f"Warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
