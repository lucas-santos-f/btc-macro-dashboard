"""Series registry + one adapter per data source.

New metric: add an entry to SERIES. New source: write a fetch_<source>(ids, start, end)
that returns a date-indexed DataFrame, one column per id, and register it in ADAPTERS.

Series fields:
  key       internal id, becomes the key in data.json
  label     what the dashboard shows (the UI is in Portuguese)
  group     see GROUPS
  axis      "indexed" -> top panel, rebased to 100, log scale
            "native"  -> bottom panel, own scale, because the level itself means
                         something (MVRV = 1, Fear & Greed = 50, rates at 4%)
  source    key in ADAPTERS
  id        series id at the source
  color / prefix / suffix / decimals / scale   formatting
  default   checked when the dashboard opens
"""

import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (local dashboard)"}
TIMEOUT = 90

GROUPS = [
    {"id": "btc",         "name": "Bitcoin"},
    {"id": "valuation",   "name": "Valuation on-chain"},
    {"id": "network",     "name": "Network"},
    {"id": "exchanges",   "name": "Exchanges"},
    {"id": "sentiment",   "name": "Sentiment"},
    {"id": "indices",     "name": "Indices"},
    {"id": "rates",       "name": "Rates and volatility"},
    {"id": "commodities", "name": "Commodities and credit"},
]

SERIES = [
    dict(key="btc", label="Bitcoin", group="btc", axis="indexed", default=True,
         source="coinmetrics", id="PriceUSD", color="#F7931A", prefix="$", decimals=0),
    dict(key="realized_price", label="Realized Price", group="valuation", axis="indexed", default=True,
         source="derived", id="realized_price", color="#FFD166", prefix="$", decimals=0),
    dict(key="mktcap", label="Market Cap", group="valuation", axis="indexed",
         source="coinmetrics", id="CapMrktCurUSD", color="#E8A33D", prefix="$", decimals=0),

    # MVRV is price / realized price by definition, a separate ratio series
    # would just be the same line under another name
    dict(key="mvrv", label="MVRV", group="valuation", axis="native", default=True,
         source="coinmetrics", id="CapMVRVCur", color="#F472B6", decimals=2),
    dict(key="roi30d", label="ROI 30d", group="valuation", axis="native",
         source="coinmetrics", id="ROI30d", color="#A78BFA", suffix="%", decimals=1, scale=100),

    dict(key="hashrate", label="Hash Rate", group="network", axis="indexed",
         source="coinmetrics", id="HashRate", color="#34D399", suffix=" TH/s", decimals=0),
    dict(key="active_addresses", label="Active addresses", group="network", axis="indexed",
         source="coinmetrics", id="AdrActCnt", color="#2DD4BF", decimals=0),
    dict(key="funded_addresses", label="Funded addresses", group="network", axis="indexed",
         source="coinmetrics", id="AdrBalCnt", color="#5EEAD4", decimals=0),
    dict(key="txs", label="Transactions", group="network", axis="indexed",
         source="coinmetrics", id="TxCnt", color="#22D3EE", decimals=0),
    dict(key="issuance_usd", label="Daily issuance", group="network", axis="indexed",
         source="coinmetrics", id="IssTotUSD", color="#0EA5E9", prefix="$", decimals=0),
    dict(key="fees_btc", label="Fees paid", group="network", axis="indexed",
         source="coinmetrics", id="FeeTotNtv", color="#38BDF8", suffix=" BTC", decimals=2),
    dict(key="difficulty", label="Difficulty", group="network", axis="indexed",
         source="blockchain", id="difficulty", color="#4ADE80", decimals=0),
    dict(key="utxos", label="UTXOs", group="network", axis="indexed",
         source="blockchain", id="utxo-count", color="#86EFAC", decimals=0),
    dict(key="miner_revenue", label="Miner revenue", group="network", axis="indexed",
         source="blockchain", id="miners-revenue", color="#16A34A", prefix="$", decimals=0),

    dict(key="exchange_balance", label="Exchange balance", group="exchanges", axis="indexed", default=True,
         source="coinmetrics", id="SplyExNtv", color="#F87171", suffix=" BTC", decimals=0),
    dict(key="exchange_inflow", label="Exchange inflow", group="exchanges", axis="indexed",
         source="coinmetrics", id="FlowInExUSD", color="#FCA5A5", prefix="$", decimals=0),
    dict(key="exchange_outflow", label="Exchange outflow", group="exchanges", axis="indexed",
         source="coinmetrics", id="FlowOutExUSD", color="#FDBA74", prefix="$", decimals=0),
    dict(key="exchange_netflow", label="Net flow (in−out)", group="exchanges", axis="native",
         source="derived", id="exchange_netflow", color="#EF4444", prefix="$", decimals=0),

    dict(key="fear_greed", label="Fear & Greed", group="sentiment", axis="native", default=True,
         source="feargreed", id="fng", color="#FACC15", decimals=0),

    dict(key="dxy", label="DXY", group="indices", axis="indexed", default=True,
         source="yahoo", id="DX-Y.NYB", color="#4ADE80", decimals=2),
    dict(key="nasdaq", label="Nasdaq", group="indices", axis="indexed", default=True,
         source="yahoo", id="^IXIC", color="#60A5FA", decimals=0),
    dict(key="sp500", label="S&P 500", group="indices", axis="indexed", default=True,
         source="yahoo", id="^GSPC", color="#C084FC", decimals=0),
    dict(key="russell", label="Russell 2000", group="indices", axis="indexed",
         source="yahoo", id="^RUT", color="#818CF8", decimals=0),

    dict(key="t10", label="10-year Treasury", group="rates", axis="native",
         source="yahoo", id="^TNX", color="#94A3B8", suffix="%", decimals=2),
    dict(key="t30", label="30-year Treasury", group="rates", axis="native",
         source="yahoo", id="^TYX", color="#CBD5E1", suffix="%", decimals=2),
    dict(key="t13w", label="13-week T-bill", group="rates", axis="native",
         source="yahoo", id="^IRX", color="#64748B", suffix="%", decimals=2),
    dict(key="curve_10y_13w", label="Curve 10y − 13w", group="rates", axis="native",
         source="derived", id="curve_10y_13w", color="#E2E8F0", suffix="pp", decimals=2),
    dict(key="vix", label="VIX", group="rates", axis="native",
         source="yahoo", id="^VIX", color="#FB923C", decimals=1),

    dict(key="gold", label="Gold", group="commodities", axis="indexed",
         source="yahoo", id="GC=F", color="#EAB308", prefix="$", decimals=0),
    dict(key="silver", label="Silver", group="commodities", axis="indexed",
         source="yahoo", id="SI=F", color="#D4D4D8", prefix="$", decimals=2),
    dict(key="copper", label="Copper", group="commodities", axis="indexed",
         source="yahoo", id="HG=F", color="#EA580C", prefix="$", decimals=2),
    dict(key="oil", label="WTI crude", group="commodities", axis="indexed",
         source="yahoo", id="CL=F", color="#78716C", prefix="$", decimals=2),
    dict(key="tlt", label="20y+ Treasuries (TLT)", group="commodities", axis="indexed",
         source="yahoo", id="TLT", color="#A8A29E", prefix="$", decimals=2),
    dict(key="hyg", label="High Yield (HYG)", group="commodities", axis="indexed",
         source="yahoo", id="HYG", color="#D6D3D1", prefix="$", decimals=2),
]


def fetch_yahoo(ids, start, end):
    import yfinance as yf
    out = {}
    for tk in ids:
        df = yf.download(tk, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            print(f"    ! {tk}: empty")
            continue
        # single ticker still comes back with MultiIndex columns on recent yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out[tk] = df["Close"].dropna()
    return pd.DataFrame(out)


def fetch_coinmetrics(ids, start, end):
    """Community API, no key. 31 BTC metrics are open on the free tier."""
    r = requests.get(
        "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
        params={"assets": "btc", "metrics": ",".join(ids), "frequency": "1d",
                "start_time": start.strftime("%Y-%m-%d"),
                "end_time": end.strftime("%Y-%m-%d"), "page_size": 10000},
        headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("time").drop(columns=["asset"], errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce")


def fetch_feargreed(ids, start, end):
    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0},
                     headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()["data"]
    s = pd.Series(
        [float(x["value"]) for x in d],
        index=pd.to_datetime([int(x["timestamp"]) for x in d], unit="s").normalize(),
    ).sort_index()
    return pd.DataFrame({"fng": s})


def fetch_blockchain(ids, start, end):
    """api.blockchain.info/charts, no key. Resampled to daily."""
    out = {}
    for name in ids:
        for attempt in range(3):
            try:
                r = requests.get(f"https://api.blockchain.info/charts/{name}",
                                 params={"timespan": "8years", "format": "json"},
                                 headers=UA, timeout=TIMEOUT)
                r.raise_for_status()
                v = r.json().get("values") or []
                if not v:
                    raise RuntimeError("no values")
                s = pd.Series([p["y"] for p in v],
                              index=pd.to_datetime([p["x"] for p in v], unit="s"))
                # some charts come intraday, keep the last point of each day
                out[name] = s.resample("D").last().dropna()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    ! {name}: {type(e).__name__}, skipping")
                else:
                    time.sleep(2)
    return pd.DataFrame(out)


ADAPTERS = {
    "yahoo": fetch_yahoo,
    "coinmetrics": fetch_coinmetrics,
    "feargreed": fetch_feargreed,
    "blockchain": fetch_blockchain,
}


def derived_series(raw: pd.DataFrame) -> dict:
    d = {}
    if {"PriceUSD", "CapMVRVCur"} <= set(raw.columns):
        # CoinMetrics pulled CapRealUSD from the free tier (403). But
        # MVRV = price / realized price, so RP = price / MVRV, both still free
        d["realized_price"] = raw["PriceUSD"] / raw["CapMVRVCur"]
    if {"FlowInExUSD", "FlowOutExUSD"} <= set(raw.columns):
        d["exchange_netflow"] = raw["FlowInExUSD"] - raw["FlowOutExUSD"]
    if {"^TNX", "^IRX"} <= set(raw.columns):
        d["curve_10y_13w"] = raw["^TNX"] - raw["^IRX"]
    return d
