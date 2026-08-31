import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import predictor


class PredictorTests(unittest.TestCase):
    def test_future_direction_has_no_lookahead_at_end(self):
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        target = predictor.future_direction(pd.Series(np.arange(1, 11), index=index), 3)
        self.assertTrue(target.iloc[:-3].eq(1).all())
        self.assertTrue(target.iloc[-3:].isna().all())

    def test_future_any_uses_only_future_window(self):
        index = pd.date_range("2020-01-01", periods=8, freq="MS")
        values = pd.Series([0, 0, 1, 0, 0, 0, 0, 0], index=index)
        target = predictor.future_any(values, 2)
        self.assertEqual(target.iloc[0], 1)
        self.assertEqual(target.iloc[2], 0)
        self.assertTrue(target.iloc[-2:].isna().all())

    def test_fitted_probabilities_are_bounded(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2000-01-01", periods=700, freq="D")
        x = rng.normal(size=700)
        features = pd.DataFrame({"x": x, "lag": pd.Series(x).shift(1).to_numpy()}, index=index)
        target = pd.Series((x + rng.normal(scale=.8, size=700) > 0).astype(float), index=index)
        result = predictor.fit_forecast(features, target, min_samples=300)
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["probabilidade"], 0)
        self.assertLessEqual(result["probabilidade"], 1)
        self.assertLessEqual(result["intervalo"][0], result["intervalo"][1])

    def test_insufficient_data_is_explicit(self):
        index = pd.date_range("2020-01-01", periods=20, freq="D")
        result = predictor.fit_forecast(pd.DataFrame({"x": range(20)}, index=index),
                                        pd.Series(np.zeros(20), index=index))
        self.assertEqual(result["status"], "dados_insuficientes")

    def test_atomic_json_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            predictor.atomic_json(path, {"ok": True, "valor": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["valor"], 1)
            self.assertFalse(list(path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
