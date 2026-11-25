"""Tests for saved models, evaluation metrics, model selection and output artifacts."""

import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data_preprocessing import (FEATURES, FIGURES_DIR, METRICS_DIR, MODELS_DIR, RESULTS_DIR,
                                    load_processed_data)
from src.evaluate import BEST_MODEL_PATH, EVAL_RESULTS_CSV, slug
from src.train import MODEL_SPECS


class TestSavedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_processed_data()
        cls.models = {n: joblib.load(MODELS_DIR / s["file"]) for n, s in MODEL_SPECS.items()}

    def test_all_models_load_with_correct_features_and_classes(self):
        for name, model in self.models.items():
            self.assertEqual(list(model.feature_names_in_), FEATURES, name)
            self.assertEqual(len(model.classes_), 22, name)

    def test_scaling_rule(self):
        self.assertFalse(MODEL_SPECS["Decision Tree"]["uses_scaled"])
        self.assertFalse(MODEL_SPECS["Random Forest"]["uses_scaled"])
        self.assertTrue(MODEL_SPECS["KNN"]["uses_scaled"])
        knn_data = self.models["KNN"]._fit_X  # KNN stores its training data
        np.testing.assert_allclose(knn_data.mean(axis=0), 0, atol=1e-9)

    def test_models_were_trained_on_training_rows_only(self):
        """No leakage: every model saw exactly 1,760 rows (the training set)."""
        self.assertEqual(self.models["KNN"].n_samples_fit_, 1760)
        self.assertEqual(self.models["Decision Tree"].tree_.n_node_samples[0], 1760)
        # Each forest tree gets a bootstrap sample: 1,760 draws WITH replacement from the training
        # set. Repeated rows are stored as weights, so the total weight at the root is 1,760.
        for tree in self.models["Random Forest"].estimators_:
            self.assertEqual(tree.tree_.weighted_n_node_samples[0], 1760)

    def test_random_state(self):
        self.assertEqual(self.models["Decision Tree"].random_state, 42)
        self.assertEqual(self.models["Random Forest"].random_state, 42)


class TestEvaluationResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_processed_data()
        cls.results = pd.read_csv(EVAL_RESULTS_CSV).set_index("model")
        cls.card = json.loads(BEST_MODEL_PATH.read_text(encoding="utf-8"))

    def _inputs(self, name):
        return self.d["X_test_scaled"] if MODEL_SPECS[name]["uses_scaled"] else self.d["X_test"]

    def test_saved_metrics_match_recomputed_metrics(self):
        for name, spec in MODEL_SPECS.items():
            model = joblib.load(MODELS_DIR / spec["file"])
            pred = model.predict(self._inputs(name))
            self.assertAlmostEqual(accuracy_score(self.d["y_test"], pred),
                                   self.results.loc[name, "test_accuracy"], places=4)
            self.assertAlmostEqual(f1_score(self.d["y_test"], pred, average="macro"),
                                   self.results.loc[name, "f1_macro"], places=4)

    def test_macro_f1_independently_from_confusion_matrix(self):
        """Recompute macro F1 by hand (no scikit-learn) from the saved confusion matrix."""
        for name in MODEL_SPECS:
            cm = pd.read_csv(METRICS_DIR / f"confusion_matrix_{slug(name)}.csv", index_col=0).values
            self.assertEqual(cm.sum(), 440)
            tp = np.diag(cm)
            precision = tp / cm.sum(axis=0)
            recall = tp / cm.sum(axis=1)
            f1 = 2 * precision * recall / (precision + recall)
            self.assertAlmostEqual(f1.mean(), self.results.loc[name, "f1_macro"], places=4)
            self.assertAlmostEqual(tp.sum() / cm.sum(), self.results.loc[name, "test_accuracy"], places=4)
            self.assertEqual(cm.sum() - tp.sum(), self.results.loc[name, "misclassified"])

    def test_generalization_gap_is_train_minus_test(self):
        r = self.results
        np.testing.assert_allclose(r["generalization_gap"], r["train_accuracy"] - r["test_accuracy"], atol=1e-4)

    def test_best_model_follows_selection_rule(self):
        ranked = self.results.sort_values(["f1_macro", "cv_mean_accuracy", "generalization_gap"],
                                          ascending=[False, False, True])
        self.assertEqual(self.card["model_name"], ranked.index[0])
        self.assertEqual(self.results.loc[self.card["model_name"], "rank"], 1)
        self.assertAlmostEqual(self.card["test_metrics"]["test_accuracy"],
                               self.results.loc[self.card["model_name"], "test_accuracy"], places=4)

    def test_training_ranges_come_from_training_data(self):
        for f in FEATURES:
            self.assertEqual(self.card["training_feature_ranges"][f]["min"], self.d["X_train"][f].min())
            self.assertEqual(self.card["training_feature_ranges"][f]["max"], self.d["X_train"][f].max())


class TestArtifactsExist(unittest.TestCase):
    def test_model_files(self):
        for name in ["decision_tree", "random_forest", "knn", "scaler", "label_encoder"]:
            self.assertTrue((MODELS_DIR / f"{name}.joblib").exists(), name)
        self.assertTrue(BEST_MODEL_PATH.exists())

    def test_metric_files(self):
        files = ["preprocessing_summary.json", "training_results.json", "training_results.csv",
                 "evaluation_results.json", "evaluation_results.csv", "per_class_f1.csv"]
        for name in MODEL_SPECS:
            files += [f"classification_report_{slug(name)}.csv", f"confusion_matrix_{slug(name)}.csv"]
        for f in files:
            self.assertTrue((METRICS_DIR / f).exists(), f)

    def test_figures_01_to_17(self):
        numbers = sorted(int(p.name[:2]) for p in FIGURES_DIR.glob("[0-9][0-9]_*.png"))
        self.assertEqual(numbers, list(range(1, 18)))

    def test_reports(self):
        for f in ["eda_report.md", "evaluation_report.md"]:
            text = (RESULTS_DIR / f).read_text(encoding="utf-8")
            self.assertGreater(len(text), 1000, f)


if __name__ == "__main__":
    unittest.main()
