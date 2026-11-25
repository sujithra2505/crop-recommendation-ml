"""Tests for Phase 3 preprocessing: split, encoding, scaling, leakage, feature order."""

import hashlib
import unittest

import numpy as np
import pandas as pd

from src.data_preprocessing import (DATA_PATH, FEATURES, TARGET, load_data, load_preprocessors,
                                    load_processed_data, run_preprocessing, scale_features)

# SHA-256 of the original CSV recorded at project start (Phase 1).
ORIGINAL_DATASET_SHA256 = "54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e"


class TestRawData(unittest.TestCase):
    def test_original_dataset_unchanged(self):
        digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, ORIGINAL_DATASET_SHA256)

    def test_load_data_columns_and_size(self):
        df = load_data()
        self.assertEqual(df.shape, (2200, 8))
        self.assertEqual(list(df.columns), FEATURES + [TARGET])


class TestSavedSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_processed_data()
        cls.encoder, cls.scaler = load_preprocessors()
        cls.raw = load_data()

    def test_shapes(self):
        self.assertEqual(self.d["X_train"].shape, (1760, 7))
        self.assertEqual(self.d["X_test"].shape, (440, 7))
        self.assertEqual(len(self.d["y_train"]), 1760)
        self.assertEqual(len(self.d["y_test"]), 440)

    def test_train_and_test_do_not_overlap(self):
        train_ids, test_ids = set(self.d["X_train"].index), set(self.d["X_test"].index)
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(len(train_ids | test_ids), 2200)

    def test_stratified_80_20_per_crop(self):
        self.assertTrue((self.d["y_train"].value_counts() == 80).all())
        self.assertTrue((self.d["y_test"].value_counts() == 20).all())

    def test_feature_order_everywhere(self):
        for key in ("X_train", "X_test", "X_train_scaled", "X_test_scaled"):
            self.assertEqual(list(self.d[key].columns), FEATURES, key)
        self.assertEqual(list(self.scaler.feature_names_in_), FEATURES)

    def test_split_is_reproducible(self):
        again = run_preprocessing(save=False)
        self.assertTrue(again["X_test"].index.equals(self.d["X_test"].index))
        self.assertTrue(again["X_train"].index.equals(self.d["X_train"].index))

    def test_unscaled_rows_match_raw_data(self):
        raw_rows = self.raw.loc[self.d["X_test"].index, FEATURES]
        np.testing.assert_allclose(self.d["X_test"].values, raw_rows.values)


class TestScalingAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_processed_data()
        cls.encoder, cls.scaler = load_preprocessors()

    def test_scaler_fitted_on_training_data_only(self):
        np.testing.assert_allclose(self.scaler.mean_, self.d["X_train"].mean().values)
        np.testing.assert_allclose(self.scaler.scale_, self.d["X_train"].std(ddof=0).values)
        full = pd.concat([self.d["X_train"], self.d["X_test"]])
        self.assertFalse(np.allclose(self.scaler.mean_, full.mean().values))

    def test_scaled_train_is_standardised(self):
        np.testing.assert_allclose(self.d["X_train_scaled"].mean().values, 0, atol=1e-9)
        np.testing.assert_allclose(self.d["X_train_scaled"].std(ddof=0).values, 1, atol=1e-9)

    def test_saved_scaled_data_equals_scaler_output(self):
        for split in ("train", "test"):
            expected = scale_features(self.scaler, self.d[f"X_{split}"])
            np.testing.assert_allclose(self.d[f"X_{split}_scaled"].values, expected.values)

    def test_scale_features_uses_column_names_not_position(self):
        shuffled = self.d["X_test"][FEATURES[::-1]]
        np.testing.assert_allclose(scale_features(self.scaler, shuffled).values,
                                   self.d["X_test_scaled"].values)


class TestLabelEncoding(unittest.TestCase):
    def test_encoder_round_trip(self):
        encoder, _ = load_preprocessors()
        d = load_processed_data()
        raw = load_data()
        self.assertEqual(len(encoder.classes_), 22)
        self.assertEqual(list(encoder.classes_), sorted(encoder.classes_))
        decoded = encoder.inverse_transform(d["y_test"])
        self.assertTrue((decoded == raw.loc[d["y_test"].index, TARGET].values).all())


if __name__ == "__main__":
    unittest.main()
