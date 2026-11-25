"""
Crop Recommendation System - Streamlit web application (Phase 7).

Run from the project root (Windows PowerShell):
    streamlit run app.py

The app does NOT train anything. It loads:
  - the best model chosen in evaluation   (models/best_model.json)
  - the saved scaler and label encoder    (models/)
  - saved metrics and charts              (results/)
and uses src/predict.py, so predictions go through exactly the same
validation and preprocessing as the rest of the project.
"""

import json

import pandas as pd
import streamlit as st

from src.data_preprocessing import (DATA_PATH, FEATURES, FIGURES_DIR, METRICS_DIR, MODELS_DIR,
                                    PROCESSED_DIR, RESULTS_DIR, TARGET)
from src.predict import (FEATURE_INFO, HARD_LIMITS, CropPredictor, InputValidationError)

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="wide")

PAGES = ["🏠 Home", "🌱 Prediction", "📊 Model Performance", "🔎 EDA & Dataset", "ℹ️ About"]

CROP_EMOJI = {
    "apple": "🍎", "banana": "🍌", "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈",
    "orange": "🍊", "mango": "🥭", "coconut": "🥥", "coffee": "☕", "maize": "🌽",
    "rice": "🌾", "pomegranate": "🍎", "papaya": "🍈", "cotton": "☁️", "jute": "🧵",
}

FEATURE_HELP = {
    "N": "Nitrogen content ratio in the soil. Helps leaf growth.",
    "P": "Phosphorus content ratio in the soil. Helps roots and flowers.",
    "K": "Potassium content ratio in the soil. Helps overall plant health.",
    "temperature": "Average air temperature in degrees Celsius.",
    "humidity": "Average relative humidity in percent (0-100).",
    "ph": "Soil pH: below 7 is acidic, 7 is neutral, above 7 is alkaline.",
    "rainfall": "Rainfall in millimetres.",
}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #1b5e20 0%, #43a047 60%, #9ccc65 100%);
       color: #fff; padding: 1.6rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 2rem;}
.hero p {margin: 0; font-size: 1.05rem; opacity: .95;}
.result-card {background: #f1f8e9; border: 2px solid #66bb6a; border-radius: 14px;
              padding: 1.3rem 1.5rem; text-align: center; margin-bottom: 1rem;}
.result-card .label {color: #33691e; font-size: .95rem; text-transform: uppercase; letter-spacing: .08em;}
.result-card .crop {color: #1b5e20; font-size: 2.4rem; font-weight: 700; margin: .2rem 0;}
.small-note {color: #5f6b5f; font-size: .85rem;}
@media (max-width: 640px) {.hero h1 {font-size: 1.5rem;} .result-card .crop {font-size: 1.9rem;}}
</style>
"""


# ---------------------------------------------------------------------------
# Cached loaders: files are read once, not on every click
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model ...")
def get_predictor() -> CropPredictor:
    return CropPredictor()


@st.cache_data
def load_json(path_str: str) -> dict:
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def show_figure(filename: str, caption: str = "") -> None:
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption or None, width="stretch")
    else:
        st.warning(f"Chart not found: results/figures/{filename}. Run `python -m src.evaluate`.")


def artifacts_ready() -> bool:
    """Check that every file the app needs exists; explain how to create missing ones."""
    needed = [MODELS_DIR / "best_model.json", MODELS_DIR / "scaler.joblib",
              MODELS_DIR / "label_encoder.joblib", METRICS_DIR / "evaluation_results.csv",
              DATA_PATH]
    missing = [p for p in needed if not p.exists()]
    if missing:
        st.error("Some required files are missing:\n\n" +
                 "\n".join(f"- `{p.name}`" for p in missing) +
                 "\n\nRun these commands from the project folder, then reload the page:\n\n"
                 "```\npython -m src.data_preprocessing\npython -m src.train\npython -m src.evaluate\n```")
        return False
    return True


def crop_label(crop: str) -> str:
    return f"{CROP_EMOJI.get(crop, '🌱')} {crop.capitalize()}"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home(card: dict, evaluation: pd.DataFrame, raw: pd.DataFrame) -> None:
    st.markdown(
        '<div class="hero"><h1>🌾 Crop Recommendation System</h1>'
        "<p>Enter your soil nutrients and local weather. A machine-learning model suggests "
        "the crop that best matches those conditions.</p></div>", unsafe_allow_html=True)

    m = card["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", f"{len(raw):,}")
    c2.metric("Crops", raw[TARGET].nunique())
    c3.metric("Input features", len(FEATURES))
    c4.metric("Test accuracy", f"{m['test_accuracy']:.2%}", help=f"{card['model_name']} on 440 unseen test samples")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("How it works")
        st.markdown(
            "1. **Data:** 2,200 records of soil and weather conditions, labelled with a suitable crop.\n"
            "2. **Preprocessing:** stratified 80/20 train/test split, scaling fitted on training data only.\n"
            "3. **Training:** three models were compared: Decision Tree, Random Forest and K-Nearest Neighbors.\n"
            f"4. **Selection:** the best model on unseen test data was **{card['model_name']}**.\n"
            "5. **Prediction:** your inputs go through the same preprocessing, then the model recommends a crop.")
        st.info("Go to **🌱 Prediction** in the sidebar to try it.")
    with right:
        st.subheader("Selected model")
        st.markdown(f"### {card['model_name']}")
        st.markdown(
            f"| Metric (test set) | Score |\n|---|---|\n"
            f"| Accuracy | {m['test_accuracy']:.4f} |\n"
            f"| Precision (macro) | {m['precision_macro']:.4f} |\n"
            f"| Recall (macro) | {m['recall_macro']:.4f} |\n"
            f"| F1-score (macro) | {m['f1_macro']:.4f} |\n"
            f"| 5-fold CV accuracy | {m['cv_mean_accuracy']:.4f} ± {m['cv_std_accuracy']:.4f} |")
        st.caption(f"Compared against: {', '.join(evaluation['model'])}")


def _fill_example(example_rows: pd.DataFrame) -> None:
    """Callback: copy a typical example's values into the input boxes."""
    crop = st.session_state["example_crop"]
    if crop in example_rows.index:
        for f in FEATURES:
            st.session_state[f"in_{f}"] = float(round(example_rows.loc[crop, f], 2))


def _clear_inputs() -> None:
    for f in FEATURES:
        st.session_state[f"in_{f}"] = None


def page_prediction(predictor: CropPredictor) -> None:
    st.header("🌱 Get a crop recommendation")
    st.write("Fill in all seven values, then press **Recommend crop**.")

    # Example values = median of each crop in the TRAINING data (never the test set).
    X_train = load_csv(str(PROCESSED_DIR / "X_train.csv")).set_index("row_id")
    y_train = load_csv(str(PROCESSED_DIR / "y_train.csv")).set_index("row_id")[TARGET]
    train = X_train.assign(crop=predictor.encoder.inverse_transform(y_train))
    examples = train.groupby("crop")[FEATURES].median()

    with st.expander("Need sample values? Load a typical example", expanded=False):
        e1, e2, e3 = st.columns([3, 1, 1])
        e1.selectbox("Typical conditions for", examples.index, key="example_crop")
        e2.button("Load example", on_click=_fill_example, args=(examples,), width="stretch")
        e3.button("Clear all", on_click=_clear_inputs, width="stretch")
        st.caption("Examples are the median values of each crop in the training data.")

    ranges = predictor.training_ranges
    with st.form("prediction_form"):
        st.markdown("**Soil nutrients**")
        cols = st.columns(3)
        for col, f in zip(cols, ["N", "P", "K"]):
            _number_box(col, f, ranges)
        st.markdown("**Climate and soil**")
        cols = st.columns(4)
        for col, f in zip(cols, ["temperature", "humidity", "ph", "rainfall"]):
            _number_box(col, f, ranges)
        submitted = st.form_submit_button("🌾 Recommend crop", type="primary", width="stretch")

    if not submitted:
        return

    inputs = {f: st.session_state.get(f"in_{f}") for f in FEATURES}
    try:
        result = predictor.predict(inputs)
    except InputValidationError as e:
        st.error("Please fix the following and try again:\n\n" + "\n".join(f"- {msg}" for msg in e.errors))
        return

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f'<div class="result-card"><div class="label">Recommended crop</div>'
                    f'<div class="crop">{crop_label(result.crop)}</div>'
                    f'<div class="small-note">Model: {result.model_name}</div></div>',
                    unsafe_allow_html=True)
        for w in result.warnings:
            st.warning(w)
    with right:
        if result.confidence is not None and result.top_k:
            st.markdown("**Top recommendations**")
            for rank, (crop, p) in enumerate(result.top_k, 1):
                st.write(f"{rank}. {crop_label(crop)}: **{p:.0%}**")
                st.progress(min(max(p, 0.0), 1.0))
            st.caption(
                "**What does the percentage mean?** The Random Forest is made of 100 decision trees, and the "
                "percentage is the share of trees that voted for each crop. It shows how much the model agrees "
                "with itself. It is **not a guarantee** that the crop will grow well. Crops with 0% are not listed."
                if result.model_name == "Random Forest" else
                "The percentage is the model's estimated probability. It is **not a guarantee**.")
        else:
            st.info("This model does not provide confidence scores.")

    with st.expander("Inputs used for this prediction"):
        st.dataframe(pd.DataFrame([{FEATURE_INFO[f][0]: inputs[f] for f in FEATURES}]),
                     hide_index=True, width="stretch")


def _number_box(col, f: str, ranges: dict) -> None:
    label, unit = FEATURE_INFO[f]
    r = ranges[f]
    low, high = HARD_LIMITS[f]
    col.number_input(
        f"{label} ({unit})", key=f"in_{f}", value=None, step=1.0 if f in ("N", "P", "K") else 0.1,
        format="%.0f" if f in ("N", "P", "K") else "%.2f",
        placeholder=f"typical {r['min']:.0f}-{r['max']:.0f}" if f != "ph" else f"typical {r['min']:.1f}-{r['max']:.1f}",
        help=f"{FEATURE_HELP[f]} Allowed: {low:g} to {high:g}. Training data range: "
             f"{r['min']:.2f} to {r['max']:.2f}.")


def page_performance(card: dict, evaluation: pd.DataFrame) -> None:
    st.header("📊 Model performance")
    st.write("All three models were trained on the same 1,760 training rows and tested on the same "
             "440 unseen rows. Numbers below are the actual saved results.")
    st.success(f"Selected model: **{card['model_name']}**, chosen by highest macro F1 on the test set "
               "(ties broken by cross-validation accuracy, then smaller train-test gap).")

    table = evaluation[["rank", "model", "test_accuracy", "precision_macro", "recall_macro", "f1_macro",
                        "f1_weighted", "train_accuracy", "cv_mean_accuracy", "generalization_gap",
                        "misclassified", "training_time_sec", "prediction_time_sec"]].rename(columns={
        "rank": "Rank", "model": "Model", "test_accuracy": "Test accuracy",
        "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)",
        "f1_weighted": "F1 (weighted)", "train_accuracy": "Train accuracy", "cv_mean_accuracy": "CV accuracy",
        "generalization_gap": "Gap (train−test)", "misclassified": "Errors / 440",
        "training_time_sec": "Train time (s)", "prediction_time_sec": "Predict time (s)"})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in table.columns if c not in ("Rank", "Model", "Errors / 440")})

    tab1, tab2, tab3, tab4 = st.tabs(["Comparison", "Overfitting check", "Confusion matrices", "Per-crop F1"])
    with tab1:
        show_figure("11_model_comparison_metrics.png")
        show_figure("17_training_and_prediction_time.png")
    with tab2:
        show_figure("12_train_vs_test_accuracy.png")
        st.markdown(
            "**Gap = train accuracy − test accuracy.** A big gap means the model memorised the training data "
            "instead of learning general patterns (overfitting). A gap above 0.02 is flagged.")
        for _, r in evaluation.iterrows():
            flag = "⚠️ slight overfitting" if r["overfitting"] else "✅ OK"
            st.write(f"- **{r['model']}**: gap {r['generalization_gap']:+.4f} → {flag}")
    with tab3:
        names = {"Decision Tree": "13_confusion_matrix_decision_tree.png",
                 "Random Forest": "14_confusion_matrix_random_forest.png",
                 "KNN": "15_confusion_matrix_knn.png"}
        choice = st.radio("Model", list(names), index=list(names).index(card["model_name"]), horizontal=True)
        show_figure(names[choice])
        st.caption("Rows are the true crop, columns the predicted crop. Numbers on the diagonal are correct "
                   "predictions; anything off the diagonal is a mistake.")
    with tab4:
        show_figure("16_per_class_f1_heatmap.png")

    report = RESULTS_DIR / "evaluation_report.md"
    if report.exists():
        st.download_button("Download full evaluation report (Markdown)", report.read_text(encoding="utf-8"),
                           file_name="evaluation_report.md")


def page_eda(raw: pd.DataFrame) -> None:
    st.header("🔎 Dataset and exploratory analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Crops", raw[TARGET].nunique())
    c3.metric("Missing values", int(raw.isna().sum().sum()))
    c4.metric("Duplicate rows", int(raw.duplicated().sum()))

    st.subheader("Features")
    info = pd.DataFrame({
        "Feature": FEATURES,
        "Meaning": [FEATURE_HELP[f] for f in FEATURES],
        "Unit": [FEATURE_INFO[f][1] for f in FEATURES],
        "Min": raw[FEATURES].min().values, "Mean": raw[FEATURES].mean().values,
        "Max": raw[FEATURES].max().values,
    })
    st.dataframe(info, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("Min", "Mean", "Max")})

    with st.expander("Sample rows from the dataset"):
        st.dataframe(raw.sample(10, random_state=42), width="stretch")

    st.subheader("Key findings")
    st.markdown(
        "- The data is **clean**: no missing values, duplicates or impossible values.\n"
        "- It is **perfectly balanced**: 22 crops × 100 samples each.\n"
        "- **Outliers are real crop traits**. For example, apple and grapes need very high P and K, and rice needs very high rainfall.\n"
        "- Only **P and K** are strongly correlated (r ≈ 0.74).\n"
        "- **K, humidity and P** separate crops best; **pH** separates them least.\n"
        "- Values form neat blocks per crop, so the dataset is likely **synthetic or augmented**.")

    tabs = st.tabs(["Crop distribution", "Histograms", "Boxplots", "Correlation",
                    "Crop profiles", "By crop", "Feature strength", "Scatter", "Pair plot"])
    figs = ["01_crop_distribution.png", "02_feature_histograms.png", "03_feature_boxplots.png",
            "04_correlation_heatmap.png", "05_crop_feature_means_heatmap.png",
            "06_features_by_crop_boxplots.png", "07_feature_separation_eta_squared.png",
            "08_scatter_key_feature_pairs.png", "09_pairplot_sample_crops.png"]
    for tab, fig in zip(tabs, figs):
        with tab:
            show_figure(fig)

    report = RESULTS_DIR / "eda_report.md"
    if report.exists():
        st.download_button("Download full EDA report (Markdown)", report.read_text(encoding="utf-8"),
                           file_name="eda_report.md")


def page_about(card: dict) -> None:
    st.header("ℹ️ About this project")
    st.subheader("Objective")
    st.write("Help farmers and students choose a suitable crop from soil nutrients (N, P, K, pH) and "
             "climate (temperature, humidity, rainfall) using supervised machine learning.")
    st.subheader("Methodology")
    st.markdown(
        "1. **EDA**: checked data quality, distributions, outliers, correlations and differences between crops.\n"
        "2. **Preprocessing**: label encoding; stratified 80/20 split (`random_state=42`); "
        "`StandardScaler` fitted on training data only, to avoid data leakage.\n"
        "3. **Training**: Decision Tree and Random Forest on unscaled data; KNN on scaled data.\n"
        "4. **Evaluation**: accuracy, macro and weighted precision, recall and F1, confusion matrices, "
        "5-fold cross-validation, and a train-test gap check.\n"
        f"5. **Deployment**: the best model (**{card['model_name']}**) is served in this Streamlit app.")
    st.subheader("Algorithms")
    st.markdown(
        "- **Decision Tree**: a flowchart of yes/no questions such as *Is K > 150?*\n"
        "- **Random Forest**: 100 decision trees, each trained on a random sample; the majority vote wins. "
        "This makes it more accurate and less prone to overfitting than one tree.\n"
        "- **K-Nearest Neighbors**: finds the 5 most similar training samples and takes their majority crop. "
        "It needs scaled features because it measures distances.")
    st.subheader("Technologies")
    st.write("Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Jupyter · Streamlit")
    st.subheader("Limitations")
    st.markdown(
        "- The dataset is small (2,200 rows) and appears synthetic, so real-world accuracy will likely be lower.\n"
        "- The model only knows 22 crops and the value ranges seen during training.\n"
        "- It ignores soil type, season, irrigation, market prices, pests and local expertise.\n"
        "- Confidence percentages are not guarantees of yield or success.")
    st.subheader("Future improvements")
    st.markdown(
        "- Validate on real field data from local agricultural sources.\n"
        "- Add soil type, season and region features; tune hyperparameters with cross-validation.\n"
        "- Add explanations of each prediction, such as feature importance or SHAP.\n"
        "- Fetch weather automatically from a weather API using the user's location.")


# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("🌾 Crop Recommender")
    page = st.sidebar.radio("Navigate", PAGES, key="page")
    st.sidebar.markdown("---")
    st.sidebar.caption("Machine-learning project · Decision Tree · Random Forest · KNN")

    if not artifacts_ready():
        return

    card = load_json(str(MODELS_DIR / "best_model.json"))
    evaluation = load_csv(str(METRICS_DIR / "evaluation_results.csv"))
    raw = load_csv(str(DATA_PATH))

    if page == PAGES[0]:
        page_home(card, evaluation, raw)
    elif page == PAGES[1]:
        page_prediction(get_predictor())
    elif page == PAGES[2]:
        page_performance(card, evaluation)
    elif page == PAGES[3]:
        page_eda(raw)
    else:
        page_about(card)

    st.markdown("---")
    st.caption("Recommendations are decision support only. Always combine them with local agricultural advice.")


main()
