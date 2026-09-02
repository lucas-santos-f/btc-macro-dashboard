import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_primary_series_prevents_fallback_request(self):
        out = {"nfci": pd.Series([0.1], index=[pd.Timestamp("2026-01-01")])}
        status = {"nfci": {"ok": True, "fonte": "FRED"}}
        fetcher = mock.Mock()
        predictor._apply_fallback(out, status, ["nfci"], fetcher)
        fetcher.assert_not_called()
        self.assertEqual(status["nfci"]["fonte"], "FRED")

    def test_partial_fallback_keeps_other_failure_isolated(self):
        out = {}
        status = {
            "claims": {"ok": False, "fonte": "FRED", "erro": "timeout claims"},
            "nfci": {"ok": False, "fonte": "FRED", "erro": "timeout nfci"},
        }

        def partial(keys):
            self.assertEqual(keys, ["claims", "nfci"])
            series = pd.Series([210000.0], index=[pd.Timestamp("2026-08-29")])
            return {"claims": series}, {
                "claims": {"ok": True, "fonte": "DOL", "serie": "ETA 539"},
                "nfci": {"ok": False, "fonte": "Chicago Fed", "erro": "indisponível"},
            }

        predictor._apply_fallback(out, status, ["claims", "nfci"], partial)
        self.assertIn("claims", out)
        self.assertTrue(status["claims"]["fallback"])
        self.assertNotIn("nfci", out)
        self.assertIn("timeout nfci", status["nfci"]["erro"])
        self.assertIn("indisponível", status["nfci"]["erro"])

    def test_treasury_xml_parser_normalizes_fields(self):
        xml = """<feed xmlns='http://www.w3.org/2005/Atom'
          xmlns:m='http://schemas.microsoft.com/ado/2007/08/dataservices/metadata'
          xmlns:d='http://schemas.microsoft.com/ado/2007/08/dataservices'>
          <entry><content><m:properties><d:NEW_DATE>2026-08-31T00:00:00</d:NEW_DATE>
          <d:BC_10YEAR>4.25</d:BC_10YEAR><d:BC_3MONTH>5.10</d:BC_3MONTH>
          </m:properties></content></entry></feed>"""
        rows = predictor._parse_treasury_xml(xml, {"dgs10": "BC_10YEAR", "dgs3m": "BC_3MONTH"})
        self.assertEqual(rows["dgs10"][0][1], 4.25)
        self.assertEqual(rows["dgs3m"][0][0], pd.Timestamp("2026-08-31"))

    def test_bea_rows_parser_handles_monthly_and_quarterly_periods(self):
        monthly = predictor._parse_bea_rows([{
            "LineDescription": "Personal consumption expenditures",
            "TimePeriod": "2026M07", "DataValue": "125.40",
        }], "pce_price")
        quarterly = predictor._parse_bea_rows([{
            "LineDescription": "Gross domestic product",
            "TimePeriod": "2026Q2", "DataValue": "23,456.7",
        }], "real_gdp")
        self.assertEqual(monthly.index[0], pd.Timestamp("2026-07-01"))
        self.assertEqual(quarterly.index[0], pd.Timestamp("2026-04-01"))
        self.assertEqual(quarterly.iloc[0], 23456.7)

    def test_dol_claims_parser_uses_seasonally_adjusted_national_value(self):
        response = mock.Mock()
        response.text = """<r539cyNational><week><weekEnded>08/29/2026</weekEnded>
          <InitialClaims><NSA>198,000</NSA><SA>231,000</SA></InitialClaims>
          </week></r539cyNational>"""
        response.raise_for_status.return_value = None
        with mock.patch.object(predictor.requests, "post", return_value=response):
            out, status = predictor.fetch_claims_direct(["claims"])
        self.assertEqual(out["claims"].iloc[0], 231000.0)
        self.assertTrue(status["claims"]["ok"])

    def test_walk_forward_splits_are_strictly_temporal_and_non_overlapping(self):
        splits = predictor.temporal_walk_forward_splits(
            800, min_train=300, max_folds=5, calibration_size=60, test_size=60
        )
        self.assertGreaterEqual(len(splits), 3)
        previous_test_end = 0
        for train_end, calibration_end, test_end in splits:
            self.assertLess(train_end, calibration_end)
            self.assertLess(calibration_end, test_end)
            self.assertGreaterEqual(calibration_end, previous_test_end)
            previous_test_end = test_end

    def test_fit_forecast_reports_repeated_walk_forward(self):
        rng = np.random.default_rng(7)
        index = pd.date_range("2010-01-01", periods=1000, freq="D")
        signal = rng.normal(size=len(index))
        features = pd.DataFrame({
            "signal": signal,
            "lagged": pd.Series(signal).shift(1).to_numpy(),
        }, index=index)
        target = pd.Series((signal + rng.normal(scale=1.2, size=len(index)) > .35).astype(float),
                           index=index)
        result = predictor.fit_forecast(features, target, min_samples=350)
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["walk_forward"]["dobras_validas"], 2)
        dates = [row["data"] for row in result["historico_teste"]]
        self.assertEqual(dates, sorted(set(dates)))
        self.assertLess(
            result["walk_forward"]["dobras"][0]["treino_fim"],
            result["walk_forward"]["dobras"][0]["calibracao_inicio"],
        )

    def test_rare_event_without_temporal_support_is_explicit(self):
        index = pd.date_range("2015-01-01", periods=500, freq="D")
        features = pd.DataFrame({"x": np.linspace(-1, 1, len(index))}, index=index)
        target = pd.Series(np.r_[np.zeros(494), np.ones(6)], index=index)
        result = predictor.fit_forecast(features, target, min_samples=200)
        self.assertEqual(result["status"], "dados_insuficientes")
        self.assertTrue(result["eventos_raros"]["detectado"])
        self.assertIn("minoritária", result["motivo"])



if __name__ == "__main__":
    unittest.main()
