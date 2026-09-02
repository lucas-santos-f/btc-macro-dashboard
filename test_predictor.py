import json
import shutil
import subprocess
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


    def test_period_metrics_use_only_observed_window(self):
        index = pd.date_range("2020-01-01", periods=4, freq="D")
        metrics = predictor.period_metrics(
            pd.Series([100.0, 120.0, 90.0, 110.0], index=index),
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-04"),
        )
        self.assertAlmostEqual(metrics["retorno"], .10)
        self.assertAlmostEqual(metrics["drawdown_maximo"], -.25)
        self.assertEqual(metrics["observacoes"], 4)

    def test_recession_comparisons_keep_missing_series_explicit(self):
        market_index = pd.date_range("2006-01-01", "2021-01-01", freq="D")
        markets = pd.DataFrame({"sp500": np.linspace(100, 200, len(market_index))},
                               index=market_index)
        month_index = pd.date_range("2000-01-01", "2021-01-01", freq="MS")
        monthly = pd.DataFrame({"term_spread": np.linspace(-1, 2, len(month_index))},
                               index=month_index)
        rows = predictor.recession_comparisons(monthly, markets)
        self.assertEqual([row["id"] for row in rows], ["2000_2001", "2008", "2020"])
        self.assertIn("Bitcoin", rows[1]["series_ausentes"])
        self.assertEqual(rows[1]["origem"]["recessoes"], "NBER Business Cycle Dating Committee")
        self.assertIsNotNone(rows[1]["curva_10a_3m"])

    def test_btc_cycles_are_derived_from_available_prices(self):
        index = pd.date_range("2016-07-09", "2025-01-01", freq="D")
        values = 100 * np.exp(np.linspace(0, 2, len(index)))
        markets = pd.DataFrame({"btc": values}, index=index)
        cycles = predictor.btc_cycle_comparisons(markets)
        ids = [item["id"] for item in cycles]
        self.assertIn("halving_420000", ids)
        self.assertIn("halving_630000", ids)
        self.assertTrue(next(x for x in cycles if x["id"] == "halving_420000")["completo_desde_halving"])


    def test_similar_episodes_reserve_future_horizon_and_separate_matches(self):
        index = pd.date_range("2000-01-01", periods=180, freq="MS")
        wave = np.sin(np.arange(len(index)) / 8)
        features = pd.DataFrame({"a": wave, "b": np.cos(np.arange(len(index)) / 9)}, index=index)
        recession = pd.Series(0.0, index=index)
        markets = pd.DataFrame({
            "sp500": np.linspace(100, 300, len(index)),
            "btc": np.linspace(10, 500, len(index)),
        }, index=index)
        rows = predictor.find_similar_episodes(features, recession, markets, limit=4,
                                               separation_months=12)
        self.assertGreaterEqual(len(rows), 2)
        latest_allowed = index[-1] - pd.DateOffset(months=12)
        self.assertTrue(all(pd.Timestamp(row["data"]) <= latest_allowed for row in rows))
        periods = [pd.Timestamp(row["data"]).to_period("M") for row in rows]
        for left, right in zip(periods, periods[1:]):
            self.assertGreaterEqual(abs((left - right).n), 12)

    def test_statistical_scenarios_use_existing_interval_bounds(self):
        forecasts = [{
            "id": "recessao_12m", "titulo": "Recessão", "horizonte": "12 meses",
            "status": "ok", "probabilidade": .30, "intervalo": [.20, .45],
        }]
        scenarios = predictor.build_statistical_scenarios(forecasts)
        self.assertEqual(scenarios[0]["previsoes"]["recessao_12m"]["probabilidade"], .20)
        self.assertEqual(scenarios[1]["previsoes"]["recessao_12m"]["probabilidade"], .30)
        self.assertEqual(scenarios[2]["previsoes"]["recessao_12m"]["probabilidade"], .45)
        self.assertEqual(scenarios[0]["tipo"], "faixa_estatistica")

    def test_prediction_history_is_sorted_and_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            pd.DataFrame([
                {"gerado_em": "2026-01-03T00:00:00Z", "x": .3},
                {"gerado_em": "2026-01-01T00:00:00Z", "x": .1},
                {"gerado_em": "2026-01-02T00:00:00Z", "x": .2},
            ]).to_csv(path, index=False)
            history = predictor.load_prediction_history(path, {"x": "Teste"}, limit=2)
        points = history["series"][0]["pontos"]
        self.assertEqual([point["probabilidade"] for point in points], [.2, .3])
        self.assertEqual(history["series"][0]["titulo"], "Teste")


    def test_collected_frame_rejects_future_dates_without_losing_other_series(self):
        index = pd.to_datetime(["2026-01-01", "2026-02-01", "2030-01-01"])
        frame = pd.DataFrame({"cpi": [1.0, 2.0, 999.0], "claims": [10.0, 11.0, 12.0]}, index=index)
        status = {"cpi": {"ok": True}, "claims": {"ok": True}}
        cleaned = predictor.validate_collected_frame(
            frame, status, macro=True, as_of=pd.Timestamp("2026-12-31")
        )
        self.assertEqual(cleaned["cpi"].max(), 2.0)
        self.assertEqual(status["cpi"]["validacao"]["datas_futuras_rejeitadas"], 1)
        self.assertTrue(status["claims"]["ok"])
        self.assertTrue(status["cpi"]["governanca"]["revisavel"])
        self.assertIn("bls.gov", status["cpi"]["governanca"]["calendario_oficial"])

    def test_snapshot_is_idempotent_for_same_inputs(self):
        payload = {
            "gerado_em": "2026-01-01T00:00:00Z",
            "cobertura": {"mercado_fim": "2025-12-31"},
            "previsoes": [{"id": "x", "probabilidade": .4, "ultima_observacao": "2025-12-31"}],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            self.assertTrue(predictor.append_snapshot(payload, path))
            payload["gerado_em"] = "2026-01-01T01:00:00Z"
            self.assertFalse(predictor.append_snapshot(payload, path))
            frame = pd.read_csv(path)
        self.assertEqual(len(frame), 1)
        self.assertIn("snapshot_id", frame)

    def test_atomic_json_preserves_previous_file_on_invalid_number(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            predictor.atomic_json(path, {"estado": "válido"})
            with self.assertRaises(ValueError):
                predictor.atomic_json(path, {"valor": float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["estado"], "válido")

    def test_dashboard_has_accessibility_and_mobile_contract(self):
        html = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
        css = (Path(__file__).parent / "web" / "style.css").read_text(encoding="utf-8")
        self.assertIn('name="viewport"', html)
        self.assertIn('class="skip-link"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn('aria-labelledby="predictorTitle"', html)
        self.assertIn('body[data-view="preditor"]', css)


    def test_predictor_javascript_syntax_when_node_is_available(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js não disponível neste ambiente")
        result = subprocess.run(
            [node, "--check", str(Path(__file__).parent / "web" / "predictor.js")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_local_sync_is_fast_forward_only_and_preserves_loopback_server(self):
        root = Path(__file__).parent
        script = (root / "sincronizar-painel.ps1").read_text(encoding="utf-8")
        wrapper = (root / "sincronizar-painel.bat").read_text(encoding="utf-8")
        server = (root / "serve.py").read_text(encoding="utf-8")
        self.assertIn("git status --porcelain", script)
        self.assertIn("git fetch --prune origin main", script)
        self.assertIn("git merge-base --is-ancestor HEAD origin/main", script)
        self.assertIn("git merge --ff-only origin/main", script)
        self.assertNotIn("reset --hard", script)
        self.assertIn("sincronizar-painel.ps1", wrapper)
        self.assertIn('HOST = "127.0.0.1"', server)
        self.assertIn('default=8765', server)


if __name__ == "__main__":
    unittest.main()
