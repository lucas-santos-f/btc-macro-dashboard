# btc-macro-dashboard

Local dashboard that puts Bitcoin next to on-chain metrics, US indices, rates and commodities (34 daily series, 7 years), plus a calibrated predictor for US recession, S&P 500 drawdowns and short-term market direction.

![market view](docs/market.png)

- **Trend panel**: every series is rebased to 100 at the left edge of whatever window you're looking at, recomputed on each zoom/pan. Oscillators (MVRV, Fear & Greed, rates, VIX, net exchange flow) go in a second panel, in their own scale, one at a time.
- **Stats, in the browser**, no libraries: rolling Pearson/Spearman, lead/lag, beta/R², Engle-Granger cointegration (with the ADF pre-check that tells you when the test doesn't apply), conditional forward returns, z-score, percentile, realized vol, drawdown. Correlation and regression always run on returns, not prices.
- **Predictor**: logistic regression + gradient boosting with isotonic calibration, repeated walk-forward with contiguous train/calibration/test blocks (no random splits), macro inputs lagged one month. Brier, log loss, AUC and calibration curves are all out of sample. Rare events without enough history come back as "insufficient data" instead of a made-up probability.
- **Free data only**: CoinMetrics community API, Yahoo Finance, alternative.me, blockchain.info, FRED with fallbacks to BLS, BEA, Treasury, DOL, Chicago Fed and OFR when FRED doesn't answer.

![predictor view](docs/predictor.png)

## Run

```
pip install -r requirements.txt
python serve.py             # http://127.0.0.1:8765, a data snapshot is committed
```

Refresh the data:

```
python fetch_data.py        # ~1 min, web/data.json
python predictor.py         # a few minutes, web/predictor.json + predictor_history.csv
```

Tests: `python -m unittest test_predictor.py`

## Env

Both optional.

| var | |
|---|---|
| `BLS_API_KEY` | 20 years per BLS request instead of 10 |
| `BEA_API_KEY` | enables the BEA fallback (PCE, real GDP, income) |

## Notes

- The UI is in Portuguese (I built it for myself). Code, data keys and logs are English.
- Realized Price is derived, not downloaded: CoinMetrics took `CapRealUSD` off the free tier, but RP = price / MVRV and both of those are still free.
- `serve.py` binds to 127.0.0.1 only and sends a CSP that blocks anything external; plotly is vendored in `web/`.
- Macro data uses reference periods, not release-date vintages. The backtest lags it a month to compensate and says so in the payload.
- Known issue: the index direction cards (Nasdaq / S&P 500 "retorno positivo") are currently pinned at the 1% calibration floor while reporting a high AUC on a short test window. Most likely the isotonic step overfitting the last calibration window, not a signal. Ignore those cards until it's fixed.
- Not investment advice.
