"""
Phase 4 - Model training for the Crop Recommendation project.

Trains three classifiers on the SAME saved train split (from Phase 3):
  - Decision Tree  -> unscaled features (trees don't need scaling)
  - Random Forest  -> unscaled features
  - KNN            -> scaled features (distance-based, needs equal scales)

For each model we record training time and training accuracy, save the
fitted model to models/, and save the results to results/metrics/.

The test set is NOT used here. It is kept untouched for Phase 5 (evaluation).

Run from the project root (after `python -m src.data_preprocessing`):
    python -m src.train
"""

import json
import statistics
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (FIGURES_DIR, METRICS_DIR, MODELS_DIR,
                                    PROJECT_ROOT, RANDOM_STATE, load_processed_data)

# ---------------------------------------------------------------------------
# Model registry: one place that says which models exist, how they are built,
# where they are saved, and whether they need scaled input.
# evaluate.py, predict.py and app.py reuse this so they always feed each
# model the right kind of data.
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "Decision Tree": {
        "builder": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "file": "decision_tree.joblib",
        "uses_scaled": False,
    },
    "Random Forest": {
        # 100 trees (scikit-learn default); each tree sees a random sample of rows/features
        "builder": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "file": "random_forest.joblib",
        "uses_scaled": False,
    },
    "KNN": {
        # Predicts by majority vote of the 5 closest training samples.
        # KNN has no randomness, so random_state does not apply.
        "builder": lambda: KNeighborsClassifier(n_neighbors=5),
        "file": "knn.joblib",
        "uses_scaled": True,
    },
}

TRAINING_RESULTS_JSON = METRICS_DIR / "training_results.json"
TRAINING_RESULTS_CSV = METRICS_DIR / "training_results.csv"
TRAINING_CHART = FIGURES_DIR / "10_training_accuracy.png"


def get_training_input(model_name: str, data: dict) -> pd.DataFrame:
    """Return scaled or unscaled training features depending on the model."""
    key = "X_train_scaled" if MODEL_SPECS[model_name]["uses_scaled"] else "X_train"
    return data[key]


TIMING_REPEATS = 5  # fit several times and report the median time


def train_one_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Build, fit and time one model. Returns (fitted_model, median_seconds).

    A single timing is noisy: the very first fit in a Python session also pays
    one-off start-up costs (loading code into memory). So we fit the model
    TIMING_REPEATS times and report the median. Because random_state is fixed,
    every repeat produces the identical model; we keep the last one.
    """
    times = []
    for _ in range(TIMING_REPEATS):
        model = MODEL_SPECS[model_name]["builder"]()
        start = time.perf_counter()
        model.fit(X_train, y_train)
        times.append(time.perf_counter() - start)
    return model, statistics.median(times)


def train_all_models(data: dict) -> tuple[dict, pd.DataFrame]:
    """Train every model in MODEL_SPECS, save it, and collect results."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y_train = data["y_train"]
    models, rows = {}, []

    for name, spec in MODEL_SPECS.items():
        X_train = get_training_input(name, data)
        model, seconds = train_one_model(name, X_train, y_train)

        # Training accuracy = how well the model fits the data it learned from.
        # It is NOT a measure of real-world performance (that needs the test set).
        train_acc = accuracy_score(y_train, model.predict(X_train))

        model_path = MODELS_DIR / spec["file"]
        joblib.dump(model, model_path)

        models[name] = model
        rows.append({
            "model": name,
            "input_data": "scaled" if spec["uses_scaled"] else "unscaled",
            "train_rows": len(X_train),
            "training_time_sec": round(seconds, 4),  # median of TIMING_REPEATS fits
            "train_accuracy": round(train_acc, 4),
            "model_file": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "hyperparameters": {k: v for k, v in model.get_params().items()
                                if k in ("random_state", "n_estimators", "n_neighbors",
                                         "max_depth", "criterion", "weights", "metric")},
        })
        print(f"  {name:<14} trained in {seconds:.4f}s | train accuracy = {train_acc:.4f}")

    return models, pd.DataFrame(rows)


def save_results(results: pd.DataFrame) -> None:
    """Save training results as JSON (full detail) and CSV (easy to open in Excel)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2)
    results.drop(columns=["hyperparameters"]).to_csv(TRAINING_RESULTS_CSV, index=False)


def plot_training_accuracy(results: pd.DataFrame) -> None:
    """Bar chart of training accuracy, with training time shown on each bar."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(results["model"], results["train_accuracy"],
                  color=["#4C72B0", "#55A868", "#C44E52"])
    for bar, acc, sec in zip(bars, results["train_accuracy"], results["training_time_sec"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.4f}\n({sec:.3f}s)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.9, 1.02)  # zoomed in: all values are high, differences are small
    ax.set_ylabel("Training accuracy (axis starts at 0.90)")
    ax.set_title(f"Training accuracy by model\n(median training time of {TIMING_REPEATS} runs in brackets)")
    fig.tight_layout()
    fig.savefig(TRAINING_CHART, dpi=120)
    plt.close(fig)


def main() -> None:
    print("Loading saved train split from data/processed/ ...")
    data = load_processed_data()
    print(f"  X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}")

    print("Training models ...")
    _, results = train_all_models(data)
    save_results(results)
    plot_training_accuracy(results)

    print("\nTraining summary:")
    print(results.drop(columns=["hyperparameters", "model_file"]).to_string(index=False))
    print(f"\nSaved results -> {TRAINING_RESULTS_JSON.relative_to(PROJECT_ROOT)}, "
          f"{TRAINING_RESULTS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved chart   -> {TRAINING_CHART.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
