"""Tests for the Streamlit app using Streamlit's built-in AppTest (no browser needed)."""

import json
import unittest

from streamlit.testing.v1 import AppTest

from src.data_preprocessing import MODELS_DIR, PROJECT_ROOT

# Note: Streamlit prints a harmless "missing ScriptRunContext" warning while these tests run.

APP = str(PROJECT_ROOT / "app.py")
RICE = {"N": 90, "P": 42, "K": 43, "temperature": 20.88, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}


def open_app(page=None):
    at = AppTest.from_file(APP, default_timeout=120).run()
    if page:
        at.sidebar.radio[0].set_value(page).run()
    return at


def fill(at, values):
    for k, v in values.items():
        at.number_input(key=f"in_{k}").set_value(v)


def submit(at):
    next(b for b in at.button if "Recommend" in b.label).click().run()


def result_html(at):
    return " ".join(m.value for m in at.markdown if "result-card" in m.value and "<div" in m.value)


class TestAppPages(unittest.TestCase):
    def test_every_page_renders_without_errors(self):
        at = open_app()
        for page in at.sidebar.radio[0].options:
            at.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(at.exception), 0, page)
            self.assertEqual(len(at.error), 0, page)

    def test_home_shows_selected_model(self):
        card = json.loads((MODELS_DIR / "best_model.json").read_text(encoding="utf-8"))
        at = open_app()
        text = " ".join(m.value for m in at.markdown)
        self.assertIn(card["model_name"], text)
        self.assertEqual(len(at.metric), 4)
        self.assertEqual(at.metric[3].value, f"{card['test_metrics']['test_accuracy']:.2%}")


class TestPredictionPage(unittest.TestCase):
    def setUp(self):
        self.at = open_app("🌱 Prediction")

    def test_all_seven_inputs_present_and_empty(self):
        for f in RICE:
            self.assertIsNone(self.at.number_input(key=f"in_{f}").value)

    def test_valid_prediction(self):
        fill(self.at, RICE)
        submit(self.at)
        self.assertEqual(len(self.at.exception), 0)
        self.assertIn("Rice", result_html(self.at))
        self.assertTrue(any("not a guarantee" in c.value for c in self.at.caption))

    def test_missing_input_shows_error(self):
        fill(self.at, {k: v for k, v in RICE.items() if k != "rainfall"})
        submit(self.at)
        self.assertEqual(result_html(self.at), "")
        self.assertIn("rainfall is missing", self.at.error[0].value)

    def test_empty_form_lists_all_missing(self):
        submit(self.at)
        self.assertEqual(self.at.error[0].value.count("is missing"), 7)

    def test_invalid_value_shows_error(self):
        fill(self.at, {**RICE, "N": -10, "humidity": 120.0})
        submit(self.at)
        msg = self.at.error[0].value
        self.assertIn("N = -10", msg)
        self.assertIn("humidity = 120", msg)
        self.assertEqual(result_html(self.at), "")

    def test_out_of_training_range_shows_warning(self):
        fill(self.at, {**RICE, "N": 300})
        submit(self.at)
        self.assertTrue(any("outside the training range" in w.value for w in self.at.warning))
        self.assertNotEqual(result_html(self.at), "")

    def test_load_example_fills_inputs(self):
        self.at.selectbox(key="example_crop").set_value("apple").run()
        next(b for b in self.at.button if b.label == "Load example").click().run()
        values = {f: self.at.number_input(key=f"in_{f}").value for f in RICE}
        self.assertTrue(all(v is not None for v in values.values()))
        submit(self.at)
        self.assertIn("Apple", result_html(self.at))


if __name__ == "__main__":
    unittest.main()
