"""Tests for the prediction pipeline: validation, boundaries, consistency, decoding, CLI."""

import contextlib
import io
import json
import math
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_preprocessing import FEATURES, load_processed_data
from src.evaluate import EVAL_RESULTS_CSV
from src.predict import (HARD_LIMITS, CropPredictor, InputValidationError, main,
                         validate_inputs)

RICE = {"N": 90, "P": 42, "K": 43, "temperature": 20.88, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
APPLE = {"N": 20, "P": 134, "K": 200, "temperature": 22.6, "humidity": 92.0, "ph": 5.9, "rainfall": 112.0}


class TestValidation(unittest.TestCase):
    def test_valid_input_returns_floats_in_feature_order(self):
        clean = validate_inputs(dict(reversed(list(RICE.items()))))
        self.assertEqual(list(clean), FEATURES)
        self.assertTrue(all(isinstance(v, float) for v in clean.values()))

    def test_numeric_strings_and_numpy_types_accepted(self):
        raw = {k: str(v) for k, v in RICE.items()}
        raw["N"] = np.int64(90)
        raw["ph"] = np.float32(6.5)
        self.assertAlmostEqual(validate_inputs(raw)["ph"], 6.5, places=5)

    def test_missing_key(self):
        raw = dict(RICE)
        del raw["rainfall"]
        with self.assertRaises(InputValidationError) as ctx:
            validate_inputs(raw)
        self.assertIn("rainfall is missing", ctx.exception.errors)

    def test_missing_values_none_and_blank(self):
        for blank in (None, "", "   "):
            with self.assertRaises(InputValidationError, msg=repr(blank)):
                validate_inputs({**RICE, "ph": blank})

    def test_all_missing_reports_seven_errors(self):
        with self.assertRaises(InputValidationError) as ctx:
            validate_inputs({})
        self.assertEqual(len(ctx.exception.errors), 7)

    def test_invalid_types(self):
        for bad in ("abc", True, [1], {"x": 1}, float("nan"), float("inf"), "-inf"):
            with self.assertRaises(InputValidationError, msg=repr(bad)):
                validate_inputs({**RICE, "N": bad})

    def test_unknown_feature_rejected(self):
        with self.assertRaises(InputValidationError) as ctx:
            validate_inputs({**RICE, "soil_type": 3})
        self.assertIn("unknown input", ctx.exception.errors[0])

    def test_non_dict_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_inputs([90, 42, 43, 20, 80, 6.5, 200])

    def test_all_errors_collected_at_once(self):
        with self.assertRaises(InputValidationError) as ctx:
            validate_inputs({**RICE, "N": -1, "humidity": 101, "ph": "x"})
        self.assertEqual(len(ctx.exception.errors), 3)

    def test_boundary_values_accepted(self):
        for f, (low, high) in HARD_LIMITS.items():
            for v in (low, high):
                self.assertEqual(validate_inputs({**RICE, f: v})[f], v, f"{f}={v}")

    def test_just_outside_boundaries_rejected(self):
        for f, (low, high) in HARD_LIMITS.items():
            for v in (low - 0.01, high + 0.01):
                with self.assertRaises(InputValidationError, msg=f"{f}={v}"):
                    validate_inputs({**RICE, f: v})


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = CropPredictor()
        cls.d = load_processed_data()

    def test_loads_selected_model(self):
        self.assertEqual(self.p.model_name, self.p.card["model_name"])
        self.assertTrue(hasattr(self.p.model, "predict"))

    def test_known_examples(self):
        self.assertEqual(self.p.predict(RICE).crop, "rice")  # first row of the raw dataset
        self.assertEqual(self.p.predict(APPLE).crop, "apple")

    def test_result_structure(self):
        r = self.p.predict(RICE)
        self.assertIsInstance(r.crop, str)
        if self.p.supports_probability:
            self.assertTrue(0 < r.confidence <= 1)
            self.assertLessEqual(len(r.top_k), 3)
            self.assertEqual(r.top_k[0][0], r.crop)
            probs = [p for _, p in r.top_k]
            self.assertEqual(probs, sorted(probs, reverse=True))
            self.assertTrue(all(p > 0 for p in probs))
            self.assertLessEqual(sum(probs), 1 + 1e-9)

    def test_input_key_order_does_not_matter(self):
        shuffled = dict(reversed(list(RICE.items())))
        self.assertEqual(self.p.predict(shuffled).crop, self.p.predict(RICE).crop)
        a, b = self.p.predict(shuffled), self.p.predict(RICE)
        self.assertEqual(a.top_k, b.top_k)

    def test_pipeline_reproduces_evaluation_accuracy(self):
        """Predicting through CropPredictor on raw test rows gives the saved test accuracy."""
        pred = self.p.predict_many(self.d["X_test"])
        true = self.p.encoder.inverse_transform(self.d["y_test"])
        saved = pd.read_csv(EVAL_RESULTS_CSV).set_index("model").loc[self.p.model_name, "test_accuracy"]
        self.assertAlmostEqual((pred == true).mean(), saved, places=4)

    def test_single_and_batch_predictions_agree(self):
        rows = self.d["X_test"].head(25)
        batch = self.p.predict_many(rows)
        single = [self.p.predict(row.to_dict()).crop for _, row in rows.iterrows()]
        self.assertEqual(list(batch), single)

    def test_shuffled_columns_give_same_batch_predictions(self):
        rows = self.d["X_test"].head(50)
        np.testing.assert_array_equal(self.p.predict_many(rows), self.p.predict_many(rows[FEATURES[::-1]]))

    def test_prepare_features_unscaled_branch(self):
        X = pd.DataFrame([RICE])[FEATURES]
        if not self.p.uses_scaled:
            np.testing.assert_allclose(self.p.prepare_features(X).values, X.values)

    def test_prepare_features_scaled_branch(self):
        """If a scaled model (KNN) were selected, inputs must be scaled exactly like training data."""
        original = self.p.uses_scaled
        try:
            self.p.uses_scaled = True
            out = self.p.prepare_features(self.d["X_test"])
            np.testing.assert_allclose(out.values, self.d["X_test_scaled"].values)
        finally:
            self.p.uses_scaled = original

    def test_label_decoding(self):
        names = self.p.encoder.inverse_transform(self.p.model.classes_)
        self.assertEqual(len(set(names)), 22)
        self.assertIn(self.p.predict(RICE).crop, names)

    def test_out_of_training_range_warns_but_predicts(self):
        r = self.p.predict({**RICE, "N": 300, "temperature": 50})
        self.assertEqual(len(r.warnings), 2)
        self.assertIsInstance(r.crop, str)

    def test_in_range_has_no_warnings(self):
        self.assertEqual(self.p.predict(RICE).warnings, [])

    def test_invalid_input_raises(self):
        with self.assertRaises(InputValidationError):
            self.p.predict({**RICE, "humidity": 150})

    def test_extreme_but_valid_inputs(self):
        """All values at the hard limits: still predicts, and warns for every value outside training range."""
        lows = {f: lo for f, (lo, _) in HARD_LIMITS.items()}
        highs = {f: hi for f, (_, hi) in HARD_LIMITS.items()}
        for raw in (lows, highs):
            r = self.p.predict(raw)
            self.assertIsInstance(r.crop, str)
            expected = sum(not (self.p.training_ranges[f]["min"] <= v <= self.p.training_ranges[f]["max"])
                           for f, v in raw.items())
            self.assertEqual(len(r.warnings), expected)
            if r.confidence is not None:
                self.assertFalse(math.isnan(r.confidence))
        self.assertEqual(len(self.p.predict(highs).warnings), 7)  # every high limit is beyond training data


class TestMissingArtifacts(unittest.TestCase):
    def test_missing_model_card_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as ctx:
                CropPredictor(models_dir=empty)
            self.assertIn("python -m src.evaluate", str(ctx.exception))

    def test_mismatched_feature_list_detected(self):
        import shutil
        from src.data_preprocessing import MODELS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            for f in MODELS_DIR.glob("*"):
                if f.is_file():
                    shutil.copy(f, tmp)
            card_path = f"{tmp}/best_model.json"
            with open(card_path, encoding="utf-8") as fh:
                card = json.load(fh)
            card["features"] = list(reversed(card["features"]))
            with open(card_path, "w", encoding="utf-8") as fh:
                json.dump(card, fh)
            with self.assertRaises(ValueError):
                CropPredictor(models_dir=tmp)


class TestCommandLine(unittest.TestCase):
    def _run(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(args)
        return code, buf.getvalue()

    def test_valid_cli(self):
        args = [x for k, v in RICE.items() for x in (f"--{k}", str(v))]
        code, out = self._run(args)
        self.assertEqual(code, 0)
        self.assertIn("Recommended crop: rice", out)

    def test_invalid_cli(self):
        raw = {**RICE, "ph": "abc"}
        args = [x for k, v in raw.items() for x in (f"--{k}", str(v))]
        code, out = self._run(args)
        self.assertEqual(code, 1)
        self.assertIn("ph must be a number", out)


if __name__ == "__main__":
    unittest.main()
