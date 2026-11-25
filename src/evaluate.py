"""
Phase 5 - Model evaluation for the Crop Recommendation project.

Uses the saved models (Phase 4) and the saved, untouched test split (Phase 3).
Nothing is retrained on the test set.

For each model it computes:
  - accuracy, precision, recall, F1 (macro and weighted averages)
  - a per-crop classification report
  - a confusion matrix
  - train vs test accuracy and the generalization gap (overfitting check)
  - 5-fold cross-validation accuracy on the TRAINING set (stability check)
  - prediction time on the test set

Then it selects the best model from these actual results using a fixed,
written-down rule (see `select_best_model`) and saves:
  - models/best_model.json            (which model the app should use)
  - results/metrics/*.json / *.csv    (all numbers)
  - results/figures/11-17_*.png       (charts)
  - results/evaluation_report.md      (report generated from the numbers)

Run from the project root (after preprocessing and training):
    python -m src.evaluate
"""

import json
import statistics
import time
from datetime import date

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import (FEATURES, FIGURES_DIR, METRICS_DIR, MODELS_DIR,
                                    PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR,
                                    load_preprocessors, load_processed_data)
from src.train import MODEL_SPECS, TRAINING_RESULTS_JSON

BEST_MODEL_PATH = MODELS_DIR / "best_model.json"
EVAL_RESULTS_JSON = METRICS_DIR / "evaluation_results.json"
EVAL_RESULTS_CSV = METRICS_DIR / "evaluation_results.csv"
PER_CLASS_F1_CSV = METRICS_DIR / "per_class_f1.csv"
REPORT_PATH = RESULTS_DIR / "evaluation_report.md"

CV_FOLDS = 5
TIMING_REPEATS = 5
# If train accuracy is more than this above test accuracy, we call it overfitting.
OVERFIT_GAP_THRESHOLD = 0.02

FIG_COMPARISON = "11_model_comparison_metrics.png"
FIG_TRAIN_TEST = "12_train_vs_test_accuracy.png"
FIG_PER_CLASS = "16_per_class_f1_heatmap.png"
FIG_TIMING = "17_training_and_prediction_time.png"


def slug(name: str) -> str:
    """'Random Forest' -> 'random_forest' (for file names)."""
    return name.lower().replace(" ", "_")


def confusion_fig_name(name: str, index: int) -> str:
    return f"{13 + index}_confusion_matrix_{slug(name)}.png"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_models() -> dict:
    """Load every saved model listed in MODEL_SPECS."""
    return {name: joblib.load(MODELS_DIR / spec["file"]) for name, spec in MODEL_SPECS.items()}


def get_inputs(name: str, data: dict, split: str) -> pd.DataFrame:
    """Scaled data for KNN, unscaled for the tree models (same rule as training)."""
    suffix = "_scaled" if MODEL_SPECS[name]["uses_scaled"] else ""
    return data[f"X_{split}{suffix}"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro and weighted precision / recall / F1."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = p, r, f
    return out


def cross_validate_on_train(name: str, model, data: dict) -> np.ndarray:
    """
    5-fold stratified cross-validation on the TRAINING set only.

    For KNN the scaler is put inside a pipeline, so in every fold it is
    re-fitted on that fold's training part only. Using the already-scaled
    data here would let each validation fold influence its own scaling
    (a small data leak inside cross-validation).
    """
    if MODEL_SPECS[name]["uses_scaled"]:
        estimator = make_pipeline(StandardScaler(), clone(model))
    else:
        estimator = clone(model)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(estimator, data["X_train"], data["y_train"], cv=cv, scoring="accuracy")


def time_predictions(model, X) -> float:
    """Median time (seconds) to predict the whole test set."""
    times = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def evaluate_all(models: dict, data: dict, encoder) -> tuple[pd.DataFrame, dict, dict]:
    """Evaluate every model. Returns (summary table, reports, confusion matrices)."""
    training = {r["model"]: r for r in json.load(open(TRAINING_RESULTS_JSON, encoding="utf-8"))}
    y_train, y_test = data["y_train"], data["y_test"]
    rows, reports, matrices = [], {}, {}

    for name, model in models.items():
        X_train, X_test = get_inputs(name, data, "train"), get_inputs(name, data, "test")
        y_pred = model.predict(X_test)
        test = compute_metrics(y_test, y_pred)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        cv_scores = cross_validate_on_train(name, model, data)

        rows.append({
            "model": name,
            "train_accuracy": train_acc,
            "test_accuracy": test["accuracy"],
            "generalization_gap": train_acc - test["accuracy"],
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            **{k: v for k, v in test.items() if k != "accuracy"},
            "misclassified": int((y_pred != y_test.values).sum()),
            "test_rows": int(len(y_test)),
            "training_time_sec": training[name]["training_time_sec"],
            "prediction_time_sec": time_predictions(model, X_test),
        })
        reports[name] = classification_report(
            y_test, y_pred, labels=range(len(encoder.classes_)),
            target_names=encoder.classes_, output_dict=True, zero_division=0)
        matrices[name] = confusion_matrix(y_test, y_pred, labels=range(len(encoder.classes_)))

    summary = pd.DataFrame(rows)
    summary["overfitting"] = summary["generalization_gap"] > OVERFIT_GAP_THRESHOLD
    return summary, reports, matrices


def select_best_model(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models by a fixed rule decided before looking at the numbers:
      1. highest test macro F1 (treats all 22 crops equally)
      2. then highest cross-validation mean accuracy (more stable estimate)
      3. then smallest generalization gap (less overfitting)
      4. then fastest prediction time
    Returns the summary sorted best-first with a 'rank' column.
    """
    ranked = summary.sort_values(
        ["f1_macro", "cv_mean_accuracy", "generalization_gap", "prediction_time_sec"],
        ascending=[False, False, True, True]).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def top_confusions(matrix: np.ndarray, classes, n: int = 5) -> list[dict]:
    """Largest off-diagonal cells: which crop was mistaken for which."""
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0]
    return [{"true": classes[i], "predicted": classes[j], "count": int(off[i, j])}
            for i, j in idx[:n] if off[i, j] > 0]


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_outputs(ranked, reports, matrices, encoder, data) -> dict:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    classes = list(encoder.classes_)

    ranked.round(4).to_csv(EVAL_RESULTS_CSV, index=False)
    per_class = pd.DataFrame({name: {c: reports[name][c]["f1-score"] for c in classes}
                              for name in ranked["model"]})
    per_class.round(4).to_csv(PER_CLASS_F1_CSV, index_label="crop")

    for name in ranked["model"]:
        pd.DataFrame(reports[name]).T.round(4).to_csv(
            METRICS_DIR / f"classification_report_{slug(name)}.csv", index_label="class")
        pd.DataFrame(matrices[name], index=classes, columns=classes).to_csv(
            METRICS_DIR / f"confusion_matrix_{slug(name)}.csv", index_label="true \\ predicted")

    full = {
        "selection_rule": ["f1_macro (desc)", "cv_mean_accuracy (desc)",
                           "generalization_gap (asc)", "prediction_time_sec (asc)"],
        "overfit_gap_threshold": OVERFIT_GAP_THRESHOLD,
        "cv_folds": CV_FOLDS,
        "models": json.loads(ranked.to_json(orient="records")),
        "top_confusions": {n: top_confusions(matrices[n], classes) for n in ranked["model"]},
    }
    with open(EVAL_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)

    # Model card for the prediction pipeline and the app.
    best = ranked.iloc[0]
    spec = MODEL_SPECS[best["model"]]
    X_train = data["X_train"]
    card = {
        "model_name": best["model"],
        "model_file": spec["file"],
        "uses_scaled": spec["uses_scaled"],
        "features": FEATURES,
        "classes": classes,
        "test_metrics": {k: round(float(best[k]), 4) for k in
                         ["test_accuracy", "precision_macro", "recall_macro", "f1_macro",
                          "precision_weighted", "recall_weighted", "f1_weighted",
                          "train_accuracy", "generalization_gap", "cv_mean_accuracy",
                          "cv_std_accuracy"]},
        # Range of values seen in TRAINING data: inputs outside it get a warning in the app.
        "training_feature_ranges": {f: {"min": float(X_train[f].min()), "max": float(X_train[f].max())}
                                    for f in FEATURES},
        "selected_on": str(date.today()),
        "selection_rule": full["selection_rule"],
    }
    with open(BEST_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    return full


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
PALETTE = {"Decision Tree": "#4C72B0", "Random Forest": "#55A868", "KNN": "#C44E52"}


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(ranked):
    metrics = ["test_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "F1\n(weighted)"]
    long = ranked.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="metric", y="score", hue="model", palette=PALETTE, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.4f", fontsize=7, padding=2, rotation=90)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(min(0.9, long["score"].min() - 0.01), 1.02)
    ax.set_xlabel("")
    ax.set_ylabel(f"Score on test set (axis starts at {ax.get_ylim()[0]:.2f})")
    ax.set_title("Test-set metrics by model")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    _save(fig, FIG_COMPARISON)


def plot_train_vs_test(ranked):
    long = ranked.melt(id_vars="model", value_vars=["train_accuracy", "cv_mean_accuracy", "test_accuracy"],
                       var_name="set", value_name="accuracy")
    long["set"] = long["set"].map({"train_accuracy": "Train", "cv_mean_accuracy": "5-fold CV (train)",
                                   "test_accuracy": "Test"})
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=long, x="model", y="accuracy", hue="set",
                palette=["#8DA0CB", "#A6D854", "#FC8D62"], ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.4f", fontsize=8, padding=2)
    ax.set_ylim(min(0.9, long["accuracy"].min() - 0.01), 1.015)
    ax.set_xlabel("")
    ax.set_ylabel(f"Accuracy (axis starts at {ax.get_ylim()[0]:.2f})")
    ax.set_title("Train vs cross-validation vs test accuracy\n(gap = train − test)")
    for i, row in ranked.reset_index(drop=True).iterrows():
        ax.text(i, ax.get_ylim()[0] + 0.004, f"gap = {row['generalization_gap']:+.4f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(title="", loc="upper right", fontsize=8)
    _save(fig, FIG_TRAIN_TEST)


def plot_confusion_matrices(ranked, matrices, classes):
    for i, name in enumerate(MODEL_SPECS):  # fixed order -> stable file names
        m = matrices[name]
        acc = ranked.loc[ranked["model"] == name, "test_accuracy"].iloc[0]
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(m, annot=True, fmt="d", cmap="Greens", cbar=False, square=True,
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    annot_kws={"fontsize": 7}, mask=(m == 0))  # hide zeros for readability
        ax.set_xlabel("Predicted crop")
        ax.set_ylabel("True crop")
        ax.set_title(f"Confusion matrix - {name} (test accuracy {acc:.4f})\n"
                     "Diagonal = correct; off-diagonal = mistakes; blank = 0")
        ax.tick_params(axis="x", rotation=60)
        _save(fig, confusion_fig_name(name, i))


def plot_per_class_f1(reports, classes):
    df = pd.DataFrame({n: [reports[n][c]["f1-score"] for c in classes] for n in MODEL_SPECS},
                      index=classes)
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.8, vmax=1.0, ax=ax,
                cbar_kws={"label": "F1-score (colour scale 0.80-1.00)"})
    ax.set_title("Per-crop F1-score on the test set")
    ax.set_ylabel("Crop")
    _save(fig, FIG_PER_CLASS)


def plot_timing(ranked):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [(axes[0], "training_time_sec", "Training time (1,760 rows)"),
                           (axes[1], "prediction_time_sec", "Prediction time (440 test rows)")]:
        order = list(MODEL_SPECS)
        vals = ranked.set_index("model").loc[order, col] * 1000
        bars = ax.bar(order, vals, color=[PALETTE[m] for m in order])
        ax.bar_label(bars, labels=[f"{v:.1f} ms" for v in vals], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("milliseconds (median of 5 runs)")
        ax.margins(y=0.15)
    _save(fig, FIG_TIMING)


# ---------------------------------------------------------------------------
# Markdown report (all numbers come from the results above)
# ---------------------------------------------------------------------------
def fmt(x, d=4):
    return f"{x:.{d}f}"


def write_report(ranked, full, reports, classes):
    best = ranked.iloc[0]
    second = ranked.iloc[1]
    lines = []
    add = lines.append

    add("# Model Evaluation Report: Crop Recommendation\n")
    add("*This file is generated by `python -m src.evaluate`. Every number comes from the saved "
        "models evaluated on the saved test set.*\n")
    add("## 1. Setup\n")
    add("| Item | Value |\n|---|---|")
    add(f"| Test set | {int(best['test_rows'])} rows (20 per crop × 22 crops), never used for training |")
    add("| Training set | 1,760 rows (80 per crop) |")
    add("| Split | stratified 80/20, `random_state=42` (from Phase 3) |")
    add("| Models | Decision Tree, Random Forest (unscaled input); KNN, k = 5 (scaled input) |")
    add(f"| Cross-validation | {CV_FOLDS}-fold stratified, on the **training set only** |")
    add(f"| Overfitting rule | flagged if train accuracy − test accuracy > {OVERFIT_GAP_THRESHOLD} |\n")

    add("## 2. What the metrics mean\n")
    add("| Metric | Plain-language meaning |\n|---|---|")
    add("| **Accuracy** | Share of test samples predicted correctly. |")
    add("| **Precision** | Of the samples the model labelled as crop X, how many really were X. Low precision means false alarms. |")
    add("| **Recall** | Of the real X samples, how many the model found. Low recall means missed cases. |")
    add("| **F1-score** | Harmonic mean of precision and recall; high only if both are high. |")
    add("| **Macro average** | Average over the 22 crops, every crop counted equally. |")
    add("| **Weighted average** | Average weighted by how many samples each crop has. The test set is perfectly balanced, so macro and weighted give almost the same values here. |")
    add("| **Confusion matrix** | Table of true crop (rows) vs predicted crop (columns). The diagonal holds correct predictions. |")
    add("| **Generalization gap** | Train accuracy − test accuracy. A large positive gap means the model memorised training data (overfitting). |")
    add("| **Cross-validation (CV)** | Train/validate 5 times on different parts of the training set. Gives a more stable estimate than one test split. |\n")

    add("## 3. Results\n")
    add("### 3.1 Test-set metrics\n")
    add("| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | Errors / 440 |")
    add("|---|---|---|---|---|---|---|---|")
    for _, r in ranked.iterrows():
        add(f"| {r['rank']} | {r['model']} | {fmt(r['test_accuracy'])} | {fmt(r['precision_macro'])} | "
            f"{fmt(r['recall_macro'])} | {fmt(r['f1_macro'])} | {fmt(r['f1_weighted'])} | {r['misclassified']} |")
    add(f"\n![Model comparison](figures/{FIG_COMPARISON})\n")

    add("### 3.2 Train vs test: overfitting check\n")
    add("| Model | Train acc. | 5-fold CV acc. (mean ± std) | Test acc. | Gap (train − test) | Overfitting? |")
    add("|---|---|---|---|---|---|")
    for _, r in ranked.iterrows():
        add(f"| {r['model']} | {fmt(r['train_accuracy'])} | {fmt(r['cv_mean_accuracy'])} ± {fmt(r['cv_std_accuracy'])} | "
            f"{fmt(r['test_accuracy'])} | {r['generalization_gap']:+.4f} | {'Yes' if r['overfitting'] else 'No'} |")
    add(f"\n![Train vs test](figures/{FIG_TRAIN_TEST})\n")
    add("**How to read this:**")
    for _, r in ranked.iterrows():
        gap = r["generalization_gap"]
        if r["train_accuracy"] >= 0.9999:
            note = (f"fits the training data perfectly (100 %). On unseen data it scores {fmt(r['test_accuracy'])}, "
                    f"a gap of {gap:+.4f}.")
        else:
            note = f"does not memorise the training data ({fmt(r['train_accuracy'])}); the gap is {gap:+.4f}."
        if not r["overfitting"]:
            verdict = "This is within the threshold, so it is not flagged as overfitting."
        elif gap < 2 * OVERFIT_GAP_THRESHOLD:
            verdict = ("This is just above the threshold: **slight overfitting**. A single unpruned tree "
                       "memorises its training rows; limiting depth or using an ensemble reduces this.")
        else:
            verdict = "This is well above the threshold: clear overfitting."
        add(f"- **{r['model']}** {note} {verdict}")
    max_diff = (ranked["cv_mean_accuracy"] - ranked["test_accuracy"]).abs().max()
    add(f"- Cross-validation accuracy is within {max_diff:.4f} of the test accuracy for every model, so the "
        "single test split is not a lucky split. (In each CV fold the model trains on only 80 % of the "
        "training set, so CV accuracy can be slightly lower.)\n")

    add("### 3.3 Confusion matrices\n")
    for i, name in enumerate(MODEL_SPECS):
        add(f"**{name}**\n\n![Confusion matrix {name}](figures/{confusion_fig_name(name, i)})\n")
    add("Most common mistakes (true → predicted):\n")
    add("| Model | Mistakes |\n|---|---|")
    for name in ranked["model"]:
        conf = full["top_confusions"][name]
        txt = ", ".join(f"{c['true']} → {c['predicted']} ({c['count']})" for c in conf) or "none"
        add(f"| {name} | {txt} |")
    add("")

    add("### 3.4 Per-crop F1-score\n")
    add(f"![Per-crop F1](figures/{FIG_PER_CLASS})\n")
    add("Crops with F1 below 1.00 for at least one model:\n")
    add("| Crop | " + " | ".join(MODEL_SPECS) + " |")
    add("|---|" + "---|" * len(MODEL_SPECS))
    for c in classes:
        vals = [reports[n][c]["f1-score"] for n in MODEL_SPECS]
        if min(vals) < 0.99995:
            add(f"| {c} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    add("\nFull per-crop reports are in `results/metrics/classification_report_<model>.csv`.\n")

    add("### 3.5 Speed\n")
    add("| Model | Training time | Prediction time (440 rows) |\n|---|---|---|")
    for name in MODEL_SPECS:
        r = ranked.set_index("model").loc[name]
        add(f"| {name} | {r['training_time_sec'] * 1000:.1f} ms | {r['prediction_time_sec'] * 1000:.1f} ms |")
    add(f"\n![Timing](figures/{FIG_TIMING})\n")
    add("All models are fast enough for an interactive app. KNN trains almost instantly because it only "
        "stores the data; its work happens at prediction time.\n")

    add("## 4. Best model\n")
    add("**Selection rule** (fixed before looking at results): highest test macro F1 → "
        "highest CV accuracy → smallest generalization gap → fastest prediction.\n")
    add(f"### Selected: **{best['model']}**\n")
    add(f"- Test accuracy **{fmt(best['test_accuracy'])}** ({int(best['misclassified'])} errors out of 440)")
    add(f"- Macro F1 **{fmt(best['f1_macro'])}**, macro precision {fmt(best['precision_macro'])}, "
        f"macro recall {fmt(best['recall_macro'])}")
    add(f"- 5-fold CV accuracy {fmt(best['cv_mean_accuracy'])} ± {fmt(best['cv_std_accuracy'])}")
    add(f"- Generalization gap {best['generalization_gap']:+.4f}\n")
    df1 = best["f1_macro"] - second["f1_macro"]
    add(f"Runner-up: **{second['model']}** (macro F1 {fmt(second['f1_macro'])}, "
        f"{int(second['misclassified'])} errors). The difference is {df1:.4f} macro F1, i.e. "
        f"{int(second['misclassified']) - int(best['misclassified'])} fewer mistakes on 440 test samples.")
    if df1 < 0.005:
        add(" The margin is small, so both models are good choices; the rule above breaks the near-tie in a transparent way.")
    add("")
    add(f"The saved choice is in `models/best_model.json`; the prediction pipeline and the app read it from there.\n")

    add("## 5. Key findings\n")
    add("1. All three models score very high on unseen data. This matches the EDA: crops have distinct, "
        "well-separated feature ranges.")
    add("2. Errors concentrate on a few crop pairs with overlapping conditions (see confusion tables). "
        "Most crops are predicted perfectly.")
    add("3. Train accuracy alone is misleading: the tree models score 100 % on training data. "
        "Only the test set and cross-validation show real performance.")
    add("4. Because the test set is balanced, macro and weighted averages are nearly equal; accuracy is a fair summary metric.")
    add("5. The EDA suggested the dataset is likely synthetic/augmented. Scores this high should not be "
        "expected on noisy real-world farm data without further validation.\n")

    add("## 6. Data leakage safeguards\n")
    add("- The test set was split off before scaling, and was never used to fit the scaler, models, or select hyper-parameters.")
    add("- The scaler was fitted on the training set only.")
    add("- Cross-validation for KNN re-fits the scaler inside each fold (scikit-learn pipeline).")
    add("- Model selection uses a rule fixed in code; no settings were tuned on the test set.\n")

    add("## 7. Files\n")
    add("| File | Content |\n|---|---|")
    add("| `results/metrics/evaluation_results.json` / `.csv` | All metrics per model, ranked |")
    add("| `results/metrics/classification_report_<model>.csv` | Per-crop precision / recall / F1 |")
    add("| `results/metrics/confusion_matrix_<model>.csv` | Confusion matrices as tables |")
    add("| `results/metrics/per_class_f1.csv` | Per-crop F1 for all models |")
    add("| `models/best_model.json` | Selected model and its metrics |")
    add(f"| `results/figures/{FIG_COMPARISON[:2]}-{FIG_TIMING[:2]}_*.png` | Charts in this report |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading saved test/train split, models and encoder ...")
    data = load_processed_data()
    encoder, _ = load_preprocessors()
    models = load_models()
    classes = list(encoder.classes_)

    print("Evaluating ...")
    summary, reports, matrices = evaluate_all(models, data, encoder)
    ranked = select_best_model(summary)
    full = save_outputs(ranked, reports, matrices, encoder, data)

    print("Drawing charts ...")
    plot_metric_comparison(ranked)
    plot_train_vs_test(ranked)
    plot_confusion_matrices(ranked, matrices, classes)
    plot_per_class_f1(reports, classes)
    plot_timing(ranked)
    write_report(ranked, full, reports, classes)

    cols = ["rank", "model", "train_accuracy", "cv_mean_accuracy", "test_accuracy", "generalization_gap",
            "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "misclassified", "overfitting"]
    print("\n" + ranked[cols].round(4).to_string(index=False))
    print(f"\nBest model: {ranked.iloc[0]['model']}  -> {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report     -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
