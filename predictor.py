"""Pulls public market + macro data, trains time-aware classifiers and writes web/predictor.json.

No paid keys. FRED series fall back to the agencies themselves (BLS, BEA,
Treasury, DOL, Chicago Fed, OFR) when FRED is unreachable; the free BEA key
is optional.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUTPUT = WEB / "predictor.json"
HISTORY = ROOT / "predictor_history.csv"
PREDICTOR_SCHEMA_VERSION = 4
MODEL_VALIDATION_DESCRIPTION = (
    "Repeated walk-forward with expanding windows; in each fold, training, calibration "
    "and test are contiguous, later and non-overlapping blocks."
)
UA = {"User-Agent": "btc-macro-dashboard/1.0 public research"}

MARKETS = {
    "btc": "BTC-USD",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "russell": "^RUT",
    "tlt": "TLT",
    "hyg": "HYG",
    "gold": "GC=F",
    "oil": "CL=F",
    "t10": "^TNX",
    "t3m": "^IRX",
}

FRED = {
    "cpi": ("CPIAUCSL", "BLS", "CPI"),
    "core_cpi": ("CPILFESL", "BLS", "Core CPI"),
    "ppi": ("PPIACO", "BLS", "PPI"),
    "payroll": ("PAYEMS", "BLS", "Nonfarm payrolls"),
    "unemployment": ("UNRATE", "BLS", "Unemployment"),
    "participation": ("CIVPART", "BLS", "Participation rate"),
    "wages": ("CES0500000003", "BLS", "Average hourly earnings"),
    "jolts": ("JTSJOL", "BLS", "JOLTS openings"),
    "pce_price": ("PCEPI", "BEA", "PCE"),
    "core_pce": ("PCEPILFE", "BEA", "Core PCE"),
    "real_gdp": ("GDPC1", "BEA", "Real GDP"),
    "real_income": ("DSPIC96", "BEA", "Real disposable income"),
    "consumption": ("PCE", "BEA", "Personal consumption"),
    "claims": ("ICSA", "Department of Labor", "Initial jobless claims"),
    "dgs10": ("DGS10", "U.S. Treasury", "10-year Treasury"),
    "dgs3m": ("DGS3MO", "U.S. Treasury", "3-month Treasury"),
    "real10": ("DFII10", "U.S. Treasury", "10-year real yield"),
    "hy_spread": ("BAMLH0A0HYM2", "ICE/BofA", "High yield spread"),
    "nfci": ("NFCI", "Chicago Fed", "Financial conditions (NFCI)"),
    "recession": ("USREC", "NBER", "NBER recession"),
    "sahm_realtime": ("SAHMREALTIME", "Sahm/FRED", "Real-time Sahm rule"),
}

BLS_DIRECT = {
    "cpi": "CUSR0000SA0", "core_cpi": "CUSR0000SA0L1E",
    "ppi": "WPUFD4", "payroll": "CES0000000001",
    "unemployment": "LNS14000000", "participation": "LNS11300000",
    "wages": "CES0500000003", "jolts": "JTS000000000000000JOL",
}

BEA_DIRECT = {
    "pce_price": ("T20304", "M", "personal consumption expenditures", False),
    "core_pce": ("T20304", "M", "personal consumption expenditures excluding food and energy", True),
    "real_gdp": ("T10106", "Q", "gross domestic product", False),
    "real_income": ("T20606", "M", "disposable personal income", False),
    "consumption": ("T20600", "M", "personal consumption expenditures", False),
}

TREASURY_DIRECT = {
    "dgs10": ("daily_treasury_yield_curve", "BC_10YEAR", 1990, "10-year Treasury", "%"),
    "dgs3m": ("daily_treasury_yield_curve", "BC_3MONTH", 1990, "3-month Treasury", "%"),
    "real10": ("daily_treasury_real_yield_curve", "TC_10YEAR", 2003, "10-year real yield", "%"),
}


RELEASE_METADATA = {
    "cpi": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/cpi.htm"},
    "core_cpi": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/cpi.htm"},
    "ppi": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/ppi.htm"},
    "payroll": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/empsit.htm"},
    "unemployment": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/empsit.htm"},
    "participation": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/empsit.htm"},
    "wages": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/empsit.htm"},
    "jolts": {"revisable": True, "release_calendar": "https://www.bls.gov/schedule/news_release/jolts.htm"},
    "pce_price": {"revisable": True, "release_calendar": "https://www.bea.gov/news/schedule"},
    "core_pce": {"revisable": True, "release_calendar": "https://www.bea.gov/news/schedule"},
    "real_gdp": {"revisable": True, "release_calendar": "https://www.bea.gov/news/schedule"},
    "real_income": {"revisable": True, "release_calendar": "https://www.bea.gov/news/schedule"},
    "consumption": {"revisable": True, "release_calendar": "https://www.bea.gov/news/schedule"},
    "claims": {"revisable": True, "release_calendar": "https://oui.doleta.gov/unemploy/claims.asp"},
    "dgs10": {"revisable": False, "release_calendar": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates"},
    "dgs3m": {"revisable": False, "release_calendar": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates"},
    "real10": {"revisable": False, "release_calendar": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates"},
    "nfci": {"revisable": True, "release_calendar": "https://www.chicagofed.org/research/data/nfci/current-data"},
    "recession": {"revisable": True, "release_calendar": "https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions"},
    "sahm_realtime": {"revisable": True, "release_calendar": "https://fred.stlouisfed.org/series/SAHMREALTIME"},
}
DEFAULT_MACRO_GOVERNANCE = {
    "revisable": True,
    "date_kind": "reference/observation period, not the release timestamp",
    "realtime_vintage": False,
    "limitation": "The current source doesn't provide the vintage known at each historical date.",
}
DEFAULT_MARKET_GOVERNANCE = {
    "revisable": False,
    "date_kind": "date of the observed close",
    "realtime_vintage": True,
    "limitation": "Closes can get operational fixes from the source, but no macro revisions.",
}

TREASURY_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
DOL_CLAIMS_URL = "https://oui.doleta.gov/unemploy/wkclaims/report.asp"
CHICAGO_NFCI_URL = "https://api.data.chicagofed.org/NFCI/nfci-data-series-csv.csv"
OFR_FSI_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

RECESSIONS = [
    ("1960-05", "1961-02"), ("1969-12", "1970-11"),
    ("1973-11", "1975-03"), ("1980-01", "1980-07"),
    ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

NBER_EPISODES = [
    {
        "id": "2000_2001", "name": "2000–2001 (dot-com / 2001 recession)",
        "start": "2001-03-01", "end": "2001-11-30",
    },
    {
        "id": "2008", "name": "2008 financial crisis",
        "start": "2007-12-01", "end": "2009-06-30",
    },
    {
        "id": "2020", "name": "2020 recession",
        "start": "2020-02-01", "end": "2020-04-30",
    },
]
NBER_SOURCE_URL = "https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions"
BTC_HALVINGS = [
    ("2012-11-28", 210000), ("2016-07-09", 420000),
    ("2020-05-11", 630000), ("2024-04-20", 840000),
]



def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def safe_number(value, digits=4):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)



def validate_collected_frame(frame: pd.DataFrame, status: dict, *,
                             macro: bool, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Normalize the index, drop future dates and attach release governance to each series."""
    reference = as_of if as_of is not None else pd.Timestamp.now("UTC").tz_localize(None)
    as_of = pd.Timestamp(reference).normalize()
    cleaned = {}
    for key in frame.columns:
        series = frame[key].copy()
        index = pd.to_datetime(series.index, errors="coerce", utc=True).tz_localize(None).normalize()
        valid_index = ~pd.isna(index)
        series = pd.Series(series.to_numpy()[valid_index], index=index[valid_index], name=key)
        duplicates = int(series.index.duplicated(keep="last").sum())
        series = series[~series.index.duplicated(keep="last")].sort_index()
        future = int((series.index > as_of).sum())
        series = series.loc[series.index <= as_of].dropna()
        state = status.setdefault(key, {"ok": False, "source": "unknown"})
        base = DEFAULT_MACRO_GOVERNANCE if macro else DEFAULT_MARKET_GOVERNANCE
        metadata = {**base, **(RELEASE_METADATA.get(key, {}) if macro else {})}
        state["governance"] = metadata
        state["validation"] = {
            "chronological": True,
            "merged_duplicates": duplicates,
            "future_dates_rejected": future,
            "cutoff_date": str(as_of.date()),
        }
        if series.empty:
            state.update({"ok": False, "error": "empty after date validation"})
        else:
            state["last_date"] = str(series.index[-1].date())
            cleaned[key] = series
    return pd.DataFrame(cleaned).sort_index()


def data_governance_summary(market_status: dict, macro_status: dict) -> dict:
    states = [*market_status.values(), *macro_status.values()]
    return {
        "revisable_series": sum(bool(item.get("governance", {}).get("revisable")) for item in states),
        "series_with_release_calendar": sum(
            bool(item.get("governance", {}).get("release_calendar")) for item in states
        ),
        "future_dates_rejected": sum(
            int(item.get("validation", {}).get("future_dates_rejected", 0)) for item in states
        ),
        "merged_duplicates": sum(
            int(item.get("validation", {}).get("merged_duplicates", 0)) for item in states
        ),
        "vintage_note": (
            "Macro dates are reference periods. With no real-time vintages, backtests apply a "
            "conservative one-month lag and state the revision limitation."
        ),
    }


def fetch_markets(years: int) -> tuple[pd.DataFrame, dict]:
    start = (pd.Timestamp.now("UTC").tz_localize(None) - pd.DateOffset(years=years)).date()
    out, status = {}, {}
    for key, ticker in MARKETS.items():
        try:
            frame = yf.download(ticker, start=start, auto_adjust=True, progress=False,
                                threads=False, timeout=45)
            if frame.empty:
                raise RuntimeError("empty response")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            series = frame["Close"].dropna().astype(float)
            series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
            out[key] = series
            status[key] = {"ok": True, "source": "Yahoo Finance", "last_date": str(series.index.max().date())}
        except Exception as exc:
            status[key] = {"ok": False, "source": "Yahoo Finance", "error": str(exc)[:160]}
    return validate_collected_frame(pd.DataFrame(out), status, macro=False), status


def fetch_fred() -> tuple[pd.DataFrame, dict]:
    out, status = {}, {}

    def one(item):
        key, (series_id, source, label) = item
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            response = requests.get(url, headers=UA, timeout=(8, 18))
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            date_col = frame.columns[0]
            value_col = frame.columns[-1]
            idx = pd.to_datetime(frame[date_col], errors="coerce")
            values = pd.to_numeric(frame[value_col], errors="coerce")
            series = pd.Series(values.to_numpy(), index=idx, name=key).dropna()
            if series.empty:
                raise RuntimeError("empty response")
            return key, series, {"ok": True, "source": source, "series_id": series_id,
                                 "label": label, "last_date": str(series.index.max().date())}
        except Exception as exc:
            return key, None, {"ok": False, "source": source, "series_id": series_id,
                               "label": label, "error": str(exc)[:160]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for key, series, state in pool.map(one, FRED.items()):
            status[key] = state
            if series is not None:
                out[key] = series
    _apply_fallback(out, status, BLS_DIRECT, fetch_bls_direct)
    _apply_fallback(out, status, BEA_DIRECT, fetch_bea_direct)
    _apply_fallback(out, status, ["claims"], fetch_claims_direct)
    _apply_fallback(out, status, TREASURY_DIRECT, fetch_treasury_direct)
    _apply_fallback(out, status, ["hy_spread"], fetch_ofr_credit_direct)
    _apply_fallback(out, status, ["nfci"], fetch_nfci_direct)
    return validate_collected_frame(pd.DataFrame(out), status, macro=True), status


def _apply_fallback(out: dict, status: dict, candidates, fetcher) -> None:
    # a fallback blowing up must not take the other adapters down with it
    missing = [key for key in candidates if key not in out]
    if not missing:
        return
    try:
        direct, direct_status = fetcher(missing)
    except Exception as exc:
        direct = {}
        direct_status = {key: {"ok": False, "source": fetcher.__name__, "error": str(exc)[:180]}
                         for key in missing}
    for key in missing:
        series = direct.get(key)
        if series is not None and not series.dropna().empty:
            state = dict(direct_status.get(key, {}))
            state.update({"ok": True, "fallback": True, "primary_series": FRED[key][0]})
            out[key] = series.dropna().sort_index()
            status[key] = state
            continue
        primary = status.get(key, {"ok": False, "source": FRED[key][1], "series_id": FRED[key][0],
                                   "label": FRED[key][2]})
        fallback = direct_status.get(key, {})
        source = fallback.get("source", fetcher.__name__)
        primary_error = primary.get("error", "primary source unavailable")
        fallback_error = fallback.get("error", "fallback returned nothing")
        status[key] = {**primary, "ok": False,
                       "error": f"{primary_error} | {source}: {fallback_error}"[-360:],
                       "attempts": [primary.get("source", "FRED"), source]}


def fetch_bls_direct(keys: list[str]) -> tuple[dict, dict]:
    # BLS API without a key only takes 10 years per request (20 with one)
    ids = [BLS_DIRECT[key] for key in keys]
    reverse = {value: key for key, value in BLS_DIRECT.items()}
    points = {key: [] for key in keys}
    errors = []
    current = datetime.now().year
    registration = os.getenv("BLS_API_KEY", "").strip()
    block = 20 if registration else 10
    for start in range(1948, current + 1, block):
        end = min(start + block - 1, current)
        body = {"seriesid": ids, "startyear": str(start), "endyear": str(end)}
        if registration:
            body["registrationkey"] = registration
        try:
            response = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                                     json=body, headers=UA, timeout=(8, 25))
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError("; ".join(data.get("message") or ["BLS rejected the request"]))
            for item in data.get("Results", {}).get("series", []):
                key = reverse.get(item.get("seriesID"))
                if key not in points:
                    continue
                for row in item.get("data", []):
                    period = row.get("period", "")
                    if not period.startswith("M") or period == "M13":
                        continue
                    date = pd.Timestamp(int(row["year"]), int(period[1:]), 1)
                    value = pd.to_numeric(row.get("value"), errors="coerce")
                    if pd.notna(value):
                        points[key].append((date, float(value)))
        except Exception as exc:
            errors.append(f"{start}-{end}: {exc}")
    out, status = {}, {}
    for key in keys:
        if points[key]:
            series = pd.Series(dict(points[key]), name=key).sort_index()
            out[key] = series
            status[key] = {"ok": True, "source": "BLS API (direct)", "series_id": BLS_DIRECT[key],
                           "label": FRED[key][2], "last_date": str(series.index.max().date())}
        else:
            status[key] = {"ok": False, "source": "BLS API (direct)", "series_id": BLS_DIRECT[key],
                           "label": FRED[key][2], "error": " | ".join(errors)[-300:] or "no data"}
    return out, status


def _parse_bea_period(value: str) -> pd.Timestamp | None:
    value = str(value).strip().upper()
    if len(value) == 7 and value[4] == "M" and value[5:].isdigit():
        return pd.Timestamp(int(value[:4]), int(value[5:]), 1)
    if len(value) == 6 and value[4] == "Q" and value[5].isdigit():
        return pd.Timestamp(int(value[:4]), (int(value[5]) - 1) * 3 + 1, 1)
    return None


def _parse_bea_rows(rows: list[dict], key: str) -> pd.Series:
    _, _, wanted, contains = BEA_DIRECT[key]
    points = {}
    for row in rows:
        description = " ".join(str(row.get("LineDescription", "")).lower().split())
        matched = wanted in description if contains else description == wanted
        if not matched:
            continue
        date = _parse_bea_period(row.get("TimePeriod", ""))
        value = pd.to_numeric(str(row.get("DataValue", "")).replace(",", ""), errors="coerce")
        if date is not None and pd.notna(value):
            points[date] = float(value)
    return pd.Series(points, name=key, dtype=float).sort_index()


def fetch_bea_direct(keys: list[str]) -> tuple[dict, dict]:
    """Official NIPA fallback, only used when the free key is set."""
    api_key = os.getenv("BEA_API_KEY", "").strip()
    out, status = {}, {}
    if not api_key:
        for key in keys:
            status[key] = {"ok": False, "source": "BEA API (direct)", "series_id": BEA_DIRECT[key][0],
                           "label": FRED[key][2], "error": "BEA_API_KEY not set"}
        return out, status
    groups = {}
    for key in keys:
        table, frequency, _, _ = BEA_DIRECT[key]
        groups.setdefault((table, frequency), []).append(key)
    for (table, frequency), group_keys in groups.items():
        try:
            response = requests.get("https://apps.bea.gov/api/data/", params={
                "UserID": api_key, "method": "GetData", "datasetname": "NIPA",
                "TableName": table, "Frequency": frequency, "Year": "X", "ResultFormat": "JSON",
            }, headers=UA, timeout=(8, 30))
            response.raise_for_status()
            results = response.json().get("BEAAPI", {}).get("Results", {})
            if results.get("Error"):
                raise RuntimeError(results["Error"].get("APIErrorDescription", "BEA rejected the request"))
            rows = results.get("Data") or []
            for key in group_keys:
                series = _parse_bea_rows(rows, key)
                if series.empty:
                    raise RuntimeError(f"{key}: NIPA line not found in {table}")
                out[key] = series
                status[key] = {"ok": True, "source": "BEA API (direct)", "series_id": table,
                               "label": FRED[key][2], "last_date": str(series.index.max().date())}
        except Exception as exc:
            for key in group_keys:
                if key not in out:
                    status[key] = {"ok": False, "source": "BEA API (direct)", "series_id": table,
                                   "label": FRED[key][2], "error": str(exc)[:180]}
    return out, status


def _parse_treasury_xml(text: str, fields: dict[str, str]) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    points = {key: [] for key in fields}
    root = ET.fromstring(text)
    for properties in root.iter():
        if properties.tag.rsplit("}", 1)[-1] != "properties":
            continue
        values = {child.tag.rsplit("}", 1)[-1]: child.text for child in properties}
        date = pd.to_datetime(values.get("NEW_DATE"), errors="coerce")
        if pd.isna(date):
            continue
        for key, field in fields.items():
            value = pd.to_numeric(values.get(field), errors="coerce")
            if pd.notna(value):
                points[key].append((pd.Timestamp(date).normalize(), float(value)))
    return points


def fetch_treasury_direct(keys: list[str]) -> tuple[dict, dict]:
    # treasury.gov XML only serves one year per call
    requested = {key: TREASURY_DIRECT[key] for key in keys}
    datasets = {}
    current = datetime.now().year
    for key, (dataset, field, start, _, _) in requested.items():
        item = datasets.setdefault(dataset, {"start": start, "fields": {}})
        item["start"] = min(item["start"], start)
        item["fields"][key] = field
    tasks = [(dataset, year, info["fields"])
             for dataset, info in datasets.items() for year in range(info["start"], current + 1)]

    def one(task):
        dataset, year, fields = task
        try:
            response = requests.get(TREASURY_URL, params={"data": dataset,
                                    "field_tdr_date_value": str(year)}, headers=UA, timeout=(8, 25))
            response.raise_for_status()
            return _parse_treasury_xml(response.text, fields), None
        except Exception as exc:
            return {}, f"{dataset}/{year}: {exc}"

    points = {key: [] for key in keys}
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for batch, error in pool.map(one, tasks):
            if error:
                errors.append(error)
            for key, rows in batch.items():
                points[key].extend(rows)
    out, status = {}, {}
    for key in keys:
        dataset, field, _, label, unit = TREASURY_DIRECT[key]
        if points[key]:
            series = pd.Series(dict(points[key]), name=key).sort_index()
            out[key] = series
            status[key] = {"ok": True, "source": "U.S. Treasury XML (direct)",
                           "series_id": f"{dataset}:{field}", "label": label, "unit": unit,
                           "last_date": str(series.index.max().date()),
                           "partial_failures": len(errors)}
        else:
            status[key] = {"ok": False, "source": "U.S. Treasury XML (direct)",
                           "series_id": f"{dataset}:{field}", "label": label,
                           "error": " | ".join(errors)[-300:] or "no data"}
    return out, status


def fetch_claims_direct(keys: list[str]) -> tuple[dict, dict]:
    if "claims" not in keys:
        return {}, {}
    try:
        response = requests.post(DOL_CLAIMS_URL, data={"level": "us", "strtdate": "1967",
                                 "enddate": str(datetime.now().year), "filetype": "xml",
                                 "submit": "Submit"}, headers=UA, timeout=(8, 45))
        response.raise_for_status()
        root = ET.fromstring(response.text)
        points = {}
        for week in root.findall(".//week"):
            date = pd.to_datetime(week.findtext("weekEnded"), errors="coerce")
            raw = (week.findtext("InitialClaims/SA") or "").replace(",", "").strip()
            value = pd.to_numeric(raw, errors="coerce")
            if pd.notna(date) and pd.notna(value):
                points[pd.Timestamp(date).normalize()] = float(value)
        series = pd.Series(points, name="claims", dtype=float).sort_index()
        if series.empty:
            raise RuntimeError("ETA 539 report has no national SA claims")
        return {"claims": series}, {"claims": {"ok": True, "source": "U.S. DOL ETA (direct)",
                "series_id": "ETA 539 initial claims SA", "label": FRED["claims"][2],
                "unit": " claims", "last_date": str(series.index.max().date())}}
    except Exception as exc:
        return {}, {"claims": {"ok": False, "source": "U.S. DOL ETA (direct)",
                    "series_id": "ETA 539 initial claims SA", "label": FRED["claims"][2],
                    "error": str(exc)[:180]}}


def fetch_nfci_direct(keys: list[str]) -> tuple[dict, dict]:
    if "nfci" not in keys:
        return {}, {}
    try:
        response = requests.get(CHICAGO_NFCI_URL, headers=UA, timeout=(8, 30))
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        index = pd.to_datetime(frame["Friday_of_Week"], errors="coerce")
        values = pd.to_numeric(frame["NFCI"], errors="coerce")
        series = pd.Series(values.to_numpy(), index=index, name="nfci").dropna().sort_index()
        if series.empty:
            raise RuntimeError("NFCI csv has no rows")
        return {"nfci": series}, {"nfci": {"ok": True, "source": "Chicago Fed (direct)",
                "series_id": "NFCI", "label": FRED["nfci"][2], "unit": " index",
                "last_date": str(series.index.max().date())}}
    except Exception as exc:
        return {}, {"nfci": {"ok": False, "source": "Chicago Fed (direct)", "series_id": "NFCI",
                    "label": FRED["nfci"][2], "error": str(exc)[:180]}}


def fetch_ofr_credit_direct(keys: list[str]) -> tuple[dict, dict]:
    if "hy_spread" not in keys:
        return {}, {}
    try:
        response = requests.get(OFR_FSI_URL, headers=UA, timeout=(8, 30))
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        index = pd.to_datetime(frame["Date"], errors="coerce")
        values = pd.to_numeric(frame["Credit"], errors="coerce")
        series = pd.Series(values.to_numpy(), index=index, name="hy_spread").dropna().sort_index()
        if series.empty:
            raise RuntimeError("OFR FSI csv has no Credit column")
        return {"hy_spread": series}, {"hy_spread": {"ok": True, "source": "OFR FSI (direct)",
                "series_id": "OFR FSI:Credit", "label": "OFR credit stress (proxy)",
                "unit": " index", "proxy_for": FRED["hy_spread"][0],
                "last_date": str(series.index.max().date())}}
    except Exception as exc:
        return {}, {"hy_spread": {"ok": False, "source": "OFR FSI (direct)",
                    "series_id": "OFR FSI:Credit", "label": "OFR credit stress (proxy)",
                    "error": str(exc)[:180]}}


def load_onchain() -> pd.DataFrame:
    path = WEB / "data.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = pd.to_datetime(data["dates"])
        wanted = {"mvrv", "exchange_netflow", "exchange_balance", "fear_greed", "hashrate"}
        columns = {s["key"]: pd.to_numeric(pd.Series(s["values"]), errors="coerce").to_numpy()
                   for s in data["series"] if s["key"] in wanted}
        return pd.DataFrame(columns, index=idx)
    except Exception:
        return pd.DataFrame()


def recession_fallback(index: pd.DatetimeIndex) -> pd.Series:
    result = pd.Series(0.0, index=index)
    periods = index.to_period("M")
    for start, end in RECESSIONS:
        mask = (periods >= pd.Period(start, "M")) & (periods <= pd.Period(end, "M"))
        result.loc[mask] = 1.0
    return result


def daily_features(markets: pd.DataFrame, onchain: pd.DataFrame) -> pd.DataFrame:
    # The index is BTC's calendar (7 days a week). Each series is computed on its
    # own sessions and only then put on the calendar: done directly on the
    # calendar, weekends broke every equity window (vol20 was NaN almost always).
    features = pd.DataFrame(index=markets.index)
    for col in markets.columns:
        s = markets[col].where(markets[col] > 0).dropna()
        log_returns = np.log(s).diff()
        features[f"{col}_r1"] = log_returns.reindex(features.index)
        for window in (5, 20, 60):
            features[f"{col}_mom{window}"] = s.pct_change(window, fill_method=None).reindex(features.index)
        features[f"{col}_vol20"] = (log_returns.rolling(20).std() * math.sqrt(252)).reindex(features.index)
    if "sp500" in markets:
        sp = markets["sp500"].dropna()
        features["sp500_drawdown"] = (sp / sp.cummax() - 1).reindex(features.index)
    if {"t10", "t3m"} <= set(markets.columns):
        features["curve_10y_3m"] = markets["t10"] - markets["t3m"]
    if {"hyg", "tlt"} <= set(markets.columns):
        ratio = (markets["hyg"] / markets["tlt"]).dropna()
        features["credit_ratio"] = ratio.reindex(features.index)
        features["credit_mom20"] = ratio.pct_change(20, fill_method=None).reindex(features.index)
    # a weekend row knows Friday's close, that's not lookahead. limit=4 covers long weekends
    features = features.ffill(limit=4)
    if not onchain.empty:
        aligned = onchain.reindex(features.index).ffill()
        for col in aligned:
            features[f"onchain_{col}"] = aligned[col]
            features[f"onchain_{col}_z"] = (aligned[col] - aligned[col].rolling(252).mean()) / aligned[col].rolling(252).std()
    return features.replace([np.inf, -np.inf], np.nan)


def monthly_features(fred: pd.DataFrame, markets: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    start = min([x for x in [fred.index.min() if not fred.empty else None,
                             markets.index.min() if not markets.empty else None] if x is not None])
    end = max([x for x in [fred.index.max() if not fred.empty else None,
                           markets.index.max() if not markets.empty else None] if x is not None])
    index = pd.date_range(start=start, end=end, freq="MS")
    monthly = fred.resample("MS").last().reindex(index) if not fred.empty else pd.DataFrame(index=index)
    # a macro print only becomes a feature the month AFTER its reference month,
    # otherwise the backtest knows CPI before it was released
    released = monthly.drop(columns=["recession"], errors="ignore").shift(1)
    features = pd.DataFrame(index=index)
    for col in released.columns:
        features[f"{col}_level"] = released[col]
        features[f"{col}_chg1"] = released[col].diff()
        features[f"{col}_pct12"] = released[col].pct_change(12, fill_method=None)
    market_month = markets.resample("MS").last().reindex(index).ffill() if not markets.empty else pd.DataFrame(index=index)
    for col in market_month.columns:
        features[f"mkt_{col}_r1m"] = market_month[col].pct_change(1, fill_method=None)
        features[f"mkt_{col}_r3m"] = market_month[col].pct_change(3, fill_method=None)
        features[f"mkt_{col}_r12m"] = market_month[col].pct_change(12, fill_method=None)
    if {"dgs10", "dgs3m"} <= set(released.columns):
        features["term_spread"] = released["dgs10"] - released["dgs3m"]
    elif {"t10", "t3m"} <= set(market_month.columns):
        features["term_spread"] = market_month["t10"] - market_month["t3m"]
    if "recession" in monthly:
        recession = monthly["recession"].fillna(0).clip(0, 1)
    else:
        recession = recession_fallback(index)
    return features.replace([np.inf, -np.inf], np.nan), recession


def future_any(binary: pd.Series, horizon: int) -> pd.Series:
    values = binary.to_numpy(float)
    out = np.full(len(values), np.nan)
    for i in range(len(values) - horizon):
        out[i] = float(np.nanmax(values[i + 1:i + horizon + 1]) > 0)
    return pd.Series(out, index=binary.index)


def future_direction(series: pd.Series, horizon: int) -> pd.Series:
    # Horizon in the asset's own sessions. On the calendar index a Sunday row did
    # Monday / NaN > 1 -> False, a free "down" label on every weekend, and the
    # equity models learned to spot weekends (AUC 0.8 on daily direction, and 1%
    # on a Monday morning before the US open, which looks exactly like a Sunday).
    s = series.dropna()
    ahead = s.shift(-horizon)
    return (ahead / s > 1).astype(float).where(ahead.notna()).reindex(series.index)


def future_drawdown(series: pd.Series, horizon: int, threshold: float) -> pd.Series:
    s = series.dropna()
    values = s.to_numpy(float)
    out = np.full(len(values), np.nan)
    for i in range(len(values) - horizon):
        future_min = values[i + 1:i + horizon + 1].min()
        out[i] = float(future_min / values[i] - 1 <= threshold)
    return pd.Series(out, index=s.index).reindex(series.index)


def _models() -> tuple[Pipeline, Pipeline]:
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.35, class_weight="balanced", max_iter=3000,
                                     random_state=7)),
    ])
    boost = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=12,
                                                  learning_rate=.045, l2_regularization=2,
                                                  random_state=7)),
    ])
    return logistic, boost


def temporal_walk_forward_splits(n: int, min_train: int, max_folds: int = 5,
                                 calibration_size: int | None = None,
                                 test_size: int | None = None) -> list[tuple[int, int, int]]:
    """Expanding train/calibration/test folds, contiguous and non-overlapping."""
    if n <= min_train:
        return []
    calibration_size = calibration_size or max(12, min(90, n // 10))
    test_size = test_size or max(12, min(90, n // 10))
    possible = (n - min_train - calibration_size) // test_size
    folds = min(max_folds, max(0, possible))
    if folds == 0:
        return []
    first_test = n - folds * test_size
    result = []
    for fold in range(folds):
        calibration_end = first_test + fold * test_size
        train_end = calibration_end - calibration_size
        test_end = min(calibration_end + test_size, n)
        if train_end >= min_train and calibration_end < test_end:
            result.append((train_end, calibration_end, test_end))
    return result


def _rare_event_summary(y: pd.Series) -> dict:
    positives = int(y.sum()) if len(y) else 0
    negatives = int(len(y) - positives)
    rate = float(positives / len(y)) if len(y) else 0.0
    return {
        "detected": bool(len(y) and (positives < 30 or rate < .10)),
        "events": positives,
        "non_events": negatives,
        "rate": safe_number(rate),
        "handling": (
            "Inverse class weights in boosting, balanced logistic regression, calibration "
            "only on a window after training, and shrinkage toward the base rate."
        ),
    }


# Isotonic needs plenty of both classes. Fed a window with a handful of one class
# it learns a staircase from those few points, and everything past them lands
# on the 1% floor. Under this many cases per class it's Platt scaling instead.
MIN_ISOTONIC_CLASS = 30


class PlattCalibrator:
    method = "platt"

    def fit(self, raw, y):
        self.model = LogisticRegression(C=1.0).fit(np.asarray(raw).reshape(-1, 1), y)
        return self

    def predict(self, raw):
        return self.model.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


def _fit_calibrator(raw: np.ndarray, y) -> IsotonicRegression | PlattCalibrator:
    y = np.asarray(y, dtype=int)
    if min(int(y.sum()), int(len(y) - y.sum())) >= MIN_ISOTONIC_CLASS:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=.01, y_max=.99)
        calibrator.method = "isotonic"
        return calibrator.fit(raw, y)
    return PlattCalibrator().fit(raw, y)


def _balanced_sample_weight(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map({label: len(y) / (2 * count) for label, count in counts.items()}).to_numpy(float)


def _fit_temporal_models(train_x: pd.DataFrame, train_y: pd.Series,
                         calibration_x: pd.DataFrame, calibration_y: pd.Series):
    usable = train_x.columns[train_x.notna().any(axis=0)]
    if len(usable) == 0:
        raise ValueError("no usable feature in the train window")
    train_x = train_x[usable]
    calibration_x = calibration_x[usable]
    if train_y.nunique() < 2 or calibration_y.nunique() < 2:
        raise ValueError("train or calibration window is missing a class")
    if min(train_y.value_counts()) < 4:
        raise ValueError("fewer than 4 examples of a class in train")
    logit, boost = _models()
    logit.fit(train_x, train_y)
    boost.fit(train_x, train_y, model__sample_weight=_balanced_sample_weight(train_y))
    raw_calibration = (
        logit.predict_proba(calibration_x)[:, 1]
        + boost.predict_proba(calibration_x)[:, 1]
    ) / 2
    calibrator = _fit_calibrator(raw_calibration, calibration_y)
    return logit, boost, calibrator, usable


def _predict_ensemble(logit: Pipeline, boost: Pipeline, calibrator,
                      frame: pd.DataFrame) -> np.ndarray:
    raw = (logit.predict_proba(frame)[:, 1] + boost.predict_proba(frame)[:, 1]) / 2
    return np.asarray(calibrator.predict(raw), dtype=float)


def _latest_temporal_partition(y: pd.Series, min_train: int, base_size: int) -> int | None:
    # most recent calibration start that still has both classes in it
    start = max(min_train, len(y) - base_size)
    step = max(6, base_size // 2)
    while start > min_train and y.iloc[start:].nunique() < 2:
        start = max(min_train, start - step)
    return start if start < len(y) and y.iloc[start:].nunique() == 2 else None


def fit_forecast(features: pd.DataFrame, target: pd.Series, min_samples=180, overlap: int = 1) -> dict:
    all_features = features.replace([np.inf, -np.inf], np.nan).copy()
    y = target.reindex(all_features.index)
    valid = y.notna() & all_features.notna().sum(axis=1).gt(0)
    frame, y = all_features.loc[valid], y.loc[valid].astype(int)
    rare = _rare_event_summary(y)
    if len(frame) < min_samples or y.nunique() < 2 or min(rare["events"], rare["non_events"]) < 8:
        return {
            "status": "insufficient_data", "samples": int(len(frame)),
            "events": rare["events"], "rare_events": rare,
            "reason": "sample or minority class too small for temporal validation",
        }

    # crashes and recessions come in clusters: in 90-session blocks most windows see
    # one class only and the folds get thrown away. Rare daily targets use a year
    rate = float(y.mean())
    block = 252 if min(rate, 1 - rate) < .25 and len(frame) > 2500 else None
    splits = temporal_walk_forward_splits(len(frame), min_samples, max_folds=8 if block else 5,
                                          calibration_size=block, test_size=block)
    predictions, actuals, dates, fold_meta = [], [], [], []
    for train_end, calibration_end, test_end in splits:
        train_x, train_y = frame.iloc[:train_end], y.iloc[:train_end]
        calibration_x = frame.iloc[train_end:calibration_end]
        calibration_y = y.iloc[train_end:calibration_end]
        test_x, test_y = frame.iloc[calibration_end:test_end], y.iloc[calibration_end:test_end]
        try:
            logit, boost, calibrator, usable = _fit_temporal_models(
                train_x, train_y, calibration_x, calibration_y
            )
        except ValueError:
            continue
        fold_predictions = _predict_ensemble(logit, boost, calibrator, test_x[usable])
        predictions.extend(fold_predictions.tolist())
        actuals.extend(test_y.to_numpy(int).tolist())
        dates.extend(test_x.index.tolist())
        fold_meta.append({
            "train_start": str(train_x.index[0].date()),
            "train_end": str(train_x.index[-1].date()),
            "calibration_start": str(calibration_x.index[0].date()),
            "calibration_end": str(calibration_x.index[-1].date()),
            "test_start": str(test_x.index[0].date()),
            "test_end": str(test_x.index[-1].date()),
            "test_samples": int(len(test_x)),
            "test_events": int(test_y.sum()),
        })

    if len(fold_meta) < 2 or len(set(actuals)) < 2:
        return {
            "status": "insufficient_data", "samples": int(len(frame)),
            "events": rare["events"], "rare_events": rare,
            "reason": "fewer than two valid temporal folds with both classes",
            "walk_forward": {"valid_folds": len(fold_meta), "folds": fold_meta},
        }

    pred_oos = np.asarray(predictions, dtype=float)
    actual_oos = np.asarray(actuals, dtype=int)
    preliminary_auc = roc_auc_score(actual_oos, pred_oos)
    # overlapping labels: 270 test days of a 60-session target are about 4 separate
    # outcomes, and an AUC of 0.84 on that said "high quality" with a 97% forecast
    effective = len(actual_oos) / max(1, overlap)
    reliability = float(np.clip((preliminary_auc - .50) / .20, 0, 1)) * min(1.0, effective / 30)
    base_rate = float(y.mean())
    pred_oos = np.clip(reliability * pred_oos + (1 - reliability) * base_rate, .001, .999)

    final_calibration_size = block or max(12, min(90, len(frame) // 10))
    final_start = _latest_temporal_partition(y, min_samples, final_calibration_size)
    if final_start is None:
        return {
            "status": "insufficient_data", "samples": int(len(frame)),
            "events": rare["events"], "rare_events": rare,
            "reason": "the last window is missing a class, nothing to calibrate on",
            "walk_forward": {"valid_folds": len(fold_meta), "folds": fold_meta},
        }
    final_train_x, final_train_y = frame.iloc[:final_start], y.iloc[:final_start]
    final_calibration_x, final_calibration_y = frame.iloc[final_start:], y.iloc[final_start:]
    try:
        logit, boost, calibrator, usable = _fit_temporal_models(
            final_train_x, final_train_y, final_calibration_x, final_calibration_y
        )
    except ValueError as exc:
        return {
            "status": "insufficient_data", "samples": int(len(frame)),
            "events": rare["events"], "rare_events": rare, "reason": str(exc),
            "walk_forward": {"valid_folds": len(fold_meta), "folds": fold_meta},
        }

    current_candidates = all_features.loc[all_features.notna().sum(axis=1).gt(0)]
    if current_candidates.empty:
        return {"status": "insufficient_data", "samples": int(len(frame)),
                "events": rare["events"], "rare_events": rare,
                "reason": "no usable current observation"}
    current_x = current_candidates.iloc[[-1]][usable]
    calibrated_current = float(_predict_ensemble(logit, boost, calibrator, current_x)[0])
    probability = float(np.clip(
        reliability * calibrated_current + (1 - reliability) * base_rate, .001, .999
    ))

    brier = brier_score_loss(actual_oos, pred_oos)
    metrics = {
        "brier": safe_number(brier),
        "log_loss": safe_number(log_loss(actual_oos, pred_oos, labels=[0, 1])),
        "auc": safe_number(roc_auc_score(actual_oos, pred_oos)),
        "accuracy": safe_number(np.mean((pred_oos >= .5) == actual_oos)),
        "base_rate": safe_number(base_rate), "model_confidence": safe_number(reliability),
        "test_samples": int(len(actual_oos)), "total_samples": int(len(frame)),
        "events": rare["events"], "folds": len(fold_meta),
        "raw_auc": safe_number(preliminary_auc), "effective_samples": safe_number(effective, 1),
    }
    auc = metrics["auc"] or 0
    quality = "high" if auc >= .70 and brier <= .20 else "moderate" if auc >= .58 else "low"
    if effective < 30:
        quality = "low"
    bins = min(8, max(3, len(actual_oos) // 25))
    observed, predicted = calibration_curve(actual_oos, pred_oos, n_bins=bins, strategy="quantile")
    calibration = [{"predicted": safe_number(p), "observed": safe_number(o)}
                   for p, o in zip(predicted, observed)]
    error = min(.35, max(.05, 1.96 * math.sqrt(max(brier, .001) / max(len(actual_oos), 1))))

    names = list(usable)
    imputed = logit.named_steps["imputer"].transform(current_x)
    scaled = logit.named_steps["scale"].transform(imputed)[0]
    coef = logit.named_steps["model"].coef_[0]
    contributions = coef[:len(names)] * scaled[:len(names)]
    order = np.argsort(np.abs(contributions))[::-1][:8]
    drivers = [{"name": names[i], "impact": safe_number(contributions[i]),
                "direction": "up" if contributions[i] >= 0 else "down"} for i in order]
    history = [
        {"date": str(pd.Timestamp(date).date()), "probability": safe_number(prob), "occurred": int(actual)}
        for date, prob, actual in zip(dates[-240:], pred_oos[-240:], actual_oos[-240:])
    ]
    return {
        "status": "ok", "probability": safe_number(probability), "quality": quality,
        "interval": [safe_number(max(0, probability - error)),
                      safe_number(min(1, probability + error))],
        "metrics": metrics, "calibration": calibration, "drivers": drivers,
        "test_history": history,
        "last_observation": str(current_candidates.index[-1].date()),
        "rare_events": rare,
        "walk_forward": {
            "method": "expanding window; calibration and test contiguous and after training",
            "valid_folds": len(fold_meta), "folds": fold_meta,
            "out_of_sample": int(len(actual_oos)),
        },
        "temporal_calibration": {
            "method": calibrator.method,
            "train_end": str(final_train_x.index[-1].date()),
            "calibration_start": str(final_calibration_x.index[0].date()),
            "calibration_end": str(final_calibration_x.index[-1].date()),
        },
    }


def add_forecast(collection: list, forecast_id: str, title: str, kind: str,
                 horizon: str, result: dict, description: str) -> None:
    collection.append({"id": forecast_id, "title": title, "kind": kind,
                       "horizon": horizon, "description": description, **result})


def latest_indicator(frame: pd.DataFrame, key: str, label: str, unit="", digits=2) -> dict | None:
    if key not in frame or frame[key].dropna().empty:
        return None
    series = frame[key].dropna()
    current = series.iloc[-1]
    previous = series.iloc[-2] if len(series) > 1 else np.nan
    return {"id": key, "name": label, "value": safe_number(current, digits), "unit": unit,
            "change": safe_number(current - previous, digits), "date": str(series.index[-1].date())}



def period_metrics(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    # only what was actually observed in the window, no interpolation
    clean = series.dropna().sort_index()
    window = clean.loc[(clean.index >= start) & (clean.index <= end)]
    if len(window) < 2:
        return None
    returns = window.pct_change(fill_method=None).dropna()
    drawdown = window / window.cummax() - 1
    return {
        "observed_start": str(window.index[0].date()),
        "observed_end": str(window.index[-1].date()),
        "observations": int(len(window)),
        "return": safe_number(window.iloc[-1] / window.iloc[0] - 1),
        "max_drawdown": safe_number(drawdown.min()),
        "annualized_volatility": safe_number(returns.std() * math.sqrt(252)) if len(returns) > 1 else None,
    }


def recession_comparisons(monthly: pd.DataFrame, markets: pd.DataFrame) -> list[dict]:
    """Fixed windows around the official NBER recessions; gaps stay explicit (None), never filled."""
    output = []
    for episode in NBER_EPISODES:
        recession_start = pd.Timestamp(episode["start"])
        recession_end = pd.Timestamp(episode["end"])
        analysis_start = recession_start - pd.DateOffset(months=12)
        analysis_end = recession_end + pd.DateOffset(months=6)
        assets, missing = {}, []
        for key, label in (("sp500", "S&P 500"), ("nasdaq", "Nasdaq"), ("btc", "Bitcoin")):
            metrics = period_metrics(
                markets[key], analysis_start, analysis_end
            ) if key in markets else None
            if metrics is None:
                missing.append(label)
            assets[key] = metrics
        curve = None
        if "term_spread" in monthly:
            curve_window = monthly["term_spread"].dropna()
            curve_window = curve_window.loc[
                (curve_window.index >= analysis_start) & (curve_window.index <= analysis_end)
            ]
            if not curve_window.empty:
                before = curve_window.loc[curve_window.index <= recession_start]
                curve = {
                    "pre_recession_value": safe_number(before.iloc[-1]) if not before.empty else None,
                    "window_min": safe_number(curve_window.min()),
                    "window_max": safe_number(curve_window.max()),
                    "observed_start": str(curve_window.index[0].date()),
                    "observed_end": str(curve_window.index[-1].date()),
                }
        output.append({
            "id": episode["id"], "name": episode["name"],
            "nber_period": [str(recession_start.date()), str(recession_end.date())],
            "analysis_window": [str(analysis_start.date()), str(analysis_end.date())],
            "window_rule": "12 months before the NBER start to 6 months after the NBER end",
            "assets": assets, "curve_10y_3m": curve,
            "coverage": "complete" if not missing else "partial",
            "missing_series": missing,
            "origin": {
                "recessions": "NBER Business Cycle Dating Committee",
                "url": NBER_SOURCE_URL,
                "markets": "Yahoo Finance; only closes present in the collected coverage",
            },
        })
    return output


def btc_cycle_comparisons(markets: pd.DataFrame) -> list[dict]:
    # cycles split at the halving blocks, which are verifiable on-chain
    if "btc" not in markets or markets["btc"].dropna().empty:
        return []
    btc = markets["btc"].dropna().sort_index()
    output = []
    for index, (start_text, block) in enumerate(BTC_HALVINGS):
        start = pd.Timestamp(start_text)
        next_start = (
            pd.Timestamp(BTC_HALVINGS[index + 1][0])
            if index + 1 < len(BTC_HALVINGS) else btc.index.max()
        )
        end = min(next_start - pd.Timedelta(days=1), btc.index.max())
        metrics = period_metrics(btc, start, end)
        if metrics is None:
            continue
        observed_start = pd.Timestamp(metrics["observed_start"])
        complete_start = abs((observed_start - start).days) <= 7
        output.append({
            "id": f"halving_{block}", "name": f"Cycle after the block {block} halving",
            "start": str(start.date()), "end": str(end.date()),
            "complete_since_halving": complete_start,
            "metrics": metrics,
            "origin": "Bitcoin blockchain: public timestamp of the halving block; Yahoo Finance prices",
            "limitation": (
                None if complete_start
                else "The available price series starts after the halving; metrics only cover the observed stretch."
            ),
        })
    return output


def build_historical_benchmarks(monthly: pd.DataFrame, markets: pd.DataFrame) -> dict:
    return {
        "recessions": recession_comparisons(monthly, markets),
        "btc_cycles": btc_cycle_comparisons(markets),
        "limitations": [
            "Comparisons are descriptive, they don't imply the episode will repeat.",
            "Missing values stay null; no interpolation or price reconstruction.",
            "Recession dates follow the NBER; market windows are declared analysis rules.",
        ],
    }



def find_similar_episodes(features: pd.DataFrame, recession: pd.Series,
                          markets: pd.DataFrame, limit: int = 5,
                          separation_months: int = 12) -> list[dict]:
    """Past macro states closest to today, leaving 12 months after each so the outcome is known."""
    frame = features.replace([np.inf, -np.inf], np.nan).sort_index()
    valid_current = frame.loc[frame.notna().sum(axis=1).gt(0)]
    if valid_current.empty:
        return []
    current_date = valid_current.index[-1]
    current = valid_current.iloc[-1]
    usable = [
        column for column in frame.columns
        if pd.notna(current.get(column)) and frame[column].notna().sum() >= 60
    ]
    if len(usable) < 2:
        return []
    history_end = current_date - pd.DateOffset(months=12)
    history = frame.loc[frame.index <= history_end, usable]
    history = history.loc[history.notna().sum(axis=1) >= max(2, len(usable) // 2)]
    if history.empty:
        return []
    median = history.median()
    scale = history.std().replace(0, np.nan)
    standardized = (history.fillna(median) - median) / scale
    current_z = (current[usable].fillna(median) - median) / scale
    distances = ((standardized - current_z) ** 2).mean(axis=1).pow(.5).dropna().sort_values()

    recession_outcome = future_any(recession.reindex(frame.index).fillna(0), 12)
    monthly_market = markets.resample("MS").last().reindex(frame.index).ffill()
    selected = []
    for date, distance in distances.items():
        if any(abs((date.to_period("M") - pd.Timestamp(item["date"]).to_period("M")).n)
               < separation_months for item in selected):
            continue
        outcomes = {"recession_12m": None, "sp500_6m": None, "btc_6m": None}
        if date in recession_outcome.index and pd.notna(recession_outcome.loc[date]):
            outcomes["recession_12m"] = bool(recession_outcome.loc[date])
        for key, result_key in (("sp500", "sp500_6m"), ("btc", "btc_6m")):
            if key in monthly_market and date in monthly_market.index:
                start_value = monthly_market.loc[date, key]
                future_date = date + pd.DateOffset(months=6)
                if future_date in monthly_market.index:
                    future_value = monthly_market.loc[future_date, key]
                    if pd.notna(start_value) and pd.notna(future_value) and start_value != 0:
                        outcomes[result_key] = safe_number(future_value / start_value - 1)
        selected.append({
            "date": str(date.date()), "distance": safe_number(distance),
            "similarity": safe_number(1 / (1 + distance)),
            "compared_variables": int(history.loc[date].notna().sum()),
            "observed_outcome": outcomes,
            "note": "Standardized similarity; only dates before the held-out horizon are used.",
        })
        if len(selected) >= limit:
            break
    return selected


def build_statistical_scenarios(forecasts: list[dict]) -> list[dict]:
    # scenarios are just the interval bounds already computed, no made-up assumptions
    usable = [item for item in forecasts if item.get("status") == "ok" and item.get("interval")]
    preferred = ["recession_12m", "sp_stress_90d", "btc_up_60d"]
    chosen = [next((item for item in usable if item["id"] == key), None) for key in preferred]
    chosen = [item for item in chosen if item is not None]
    if not chosen:
        chosen = usable[:3]
    definitions = [
        ("lower", "Lower bound", 0,
         "Lower end of each forecast's statistical interval."),
        ("central", "Central estimate", None,
         "Central probability, calibrated by the walk-forward backtest."),
        ("upper", "Upper bound", 1,
         "Upper end of each forecast's statistical interval."),
    ]
    scenarios = []
    for scenario_id, name, bound, assumption in definitions:
        values = {}
        for item in chosen:
            probability = item["probability"] if bound is None else item["interval"][bound]
            values[item["id"]] = {
                "title": item["title"], "horizon": item["horizon"],
                "probability": probability,
            }
        scenarios.append({
            "id": scenario_id, "name": name, "kind": "statistical_band",
            "assumption": assumption, "forecasts": values,
            "limitation": "Not a consensus or a causal scenario, just the model's estimated uncertainty.",
        })
    return scenarios


def load_prediction_history(path: Path = HISTORY, forecast_titles: dict | None = None,
                            limit: int = 120) -> dict:
    if not path.exists():
        return {"series": [], "observations": 0}
    try:
        frame = pd.read_csv(path)
        if "generated_at" not in frame:
            return {"series": [], "observations": 0}
        frame["generated_at"] = pd.to_datetime(frame["generated_at"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["generated_at"]).sort_values("generated_at").tail(limit)
        titles = forecast_titles or {}
        series = []
        for column in frame.columns:
            if column == "generated_at" or column == "snapshot_id":
                continue
            points = []
            for date, value in zip(frame["generated_at"], pd.to_numeric(frame[column], errors="coerce")):
                if pd.notna(value):
                    points.append({"date": date.isoformat(), "probability": safe_number(value)})
            if points:
                series.append({"id": column, "title": titles.get(column, column), "points": points})
        return {"series": series, "observations": int(len(frame)), "limit": limit}
    except Exception as exc:
        return {"series": [], "observations": 0, "error": str(exc)[:180]}


def snapshot_identity(payload: dict) -> str:
    stable = {
        "coverage": payload.get("coverage", {}),
        "forecasts": {
            item["id"]: {
                "probability": item.get("probability"),
                "last_observation": item.get("last_observation"),
            }
            for item in payload.get("forecasts", [])
        },
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(name, index=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def append_snapshot(payload: dict, path: Path = HISTORY) -> bool:
    identity = snapshot_identity(payload)
    row = {"generated_at": payload["generated_at"], "snapshot_id": identity}
    for item in payload["forecasts"]:
        row[item["id"]] = item.get("probability")
    new = pd.DataFrame([row])
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty and "snapshot_id" not in old:
        value_columns = [column for column in old.columns if column != "generated_at"]
        old["snapshot_id"] = old[value_columns].apply(
            lambda values: "legacy-" + hashlib.sha256(
                json.dumps(values.fillna("").to_dict(), sort_keys=True, default=str).encode()
            ).hexdigest()[:20],
            axis=1,
        )
    existed = bool(not old.empty and identity in set(old.get("snapshot_id", [])))
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates("snapshot_id", keep="last")
    _atomic_csv(path, merged)
    return not existed


def build(years: int) -> dict:
    markets, market_status = fetch_markets(years)
    fred, fred_status = fetch_fred()
    if markets.empty:
        raise RuntimeError("no market series came back, keeping the previous output")
    onchain = load_onchain()
    daily = daily_features(markets, onchain)
    monthly, recession = monthly_features(fred, markets)
    forecasts: list[dict] = []

    for horizon in (6, 12):
        result = fit_forecast(monthly, future_any(recession, horizon), min_samples=180, overlap=horizon)
        add_forecast(forecasts, f"recession_{horizon}m", "US recession", "macro",
                     f"{horizon} months", result,
                     "Probability of at least one NBER recession month within the horizon.")

    def sessions(asset: str) -> pd.DataFrame:
        # rows where that asset actually traded, so the "current" row is its last close
        return daily.loc[daily.index.isin(markets[asset].dropna().index)]

    if "sp500" in markets:
        for days, threshold in ((30, -.10), (90, -.10), (180, -.20)):
            result = fit_forecast(sessions("sp500"), future_drawdown(markets["sp500"], days, threshold),
                                  min_samples=700, overlap=days)
            add_forecast(forecasts, f"sp_stress_{days}d", f"S&P 500: {abs(threshold):.0%} drop",
                         "stress", f"{days} sessions", result,
                         "Probability that the lowest close in the window falls past the threshold.")

    for asset, label in (("btc", "Bitcoin"), ("nasdaq", "Nasdaq"), ("sp500", "S&P 500")):
        if asset not in markets:
            continue
        for horizon in (1, 5, 20, 60):
            result = fit_forecast(sessions(asset), future_direction(markets[asset], horizon),
                                  min_samples=700, overlap=horizon)
            add_forecast(forecasts, f"{asset}_up_{horizon}d", f"{label}: positive return",
                         "direction", f"{horizon} session{'s' if horizon > 1 else ''}", result,
                         "Calibrated probability that the close at the horizon is above today's.")

    # monthly macro events vs the PREVIOUS print, not vs consensus (no free consensus feed)
    if "cpi" in fred:
        cpi_mom = fred["cpi"].resample("MS").last().pct_change(fill_method=None)
        target = (cpi_mom.shift(-1) > cpi_mom).astype(float).where(cpi_mom.shift(-1).notna())
        add_forecast(forecasts, "cpi_accelerates", "CPI accelerates", "inflation", "next release",
                     fit_forecast(monthly, target, min_samples=300),
                     "Probability that the monthly change beats the previous monthly reading.")
    if "core_pce" in fred:
        pce_mom = fred["core_pce"].resample("MS").last().pct_change(fill_method=None)
        target = (pce_mom.shift(-1) > pce_mom).astype(float).where(pce_mom.shift(-1).notna())
        add_forecast(forecasts, "pce_accelerates", "Core PCE accelerates", "inflation", "next release",
                     fit_forecast(monthly, target, min_samples=300),
                     "Probability that the monthly change beats the previous monthly reading.")
    if "payroll" in fred:
        payroll_change = fred["payroll"].resample("MS").last().diff()
        slower = (payroll_change.shift(-1) < payroll_change).astype(float).where(payroll_change.shift(-1).notna())
        negative = (payroll_change.shift(-1) < 0).astype(float).where(payroll_change.shift(-1).notna())
        add_forecast(forecasts, "payroll_slows", "Payrolls slow down", "jobs", "next release",
                     fit_forecast(monthly, slower, min_samples=300),
                     "Probability that net job creation comes in below the previous reading.")
        add_forecast(forecasts, "payroll_negative", "Negative payrolls", "jobs", "next release",
                     fit_forecast(monthly, negative, min_samples=300),
                     "Probability of net job losses; historically rare.")
    if "unemployment" in fred:
        unemp = fred["unemployment"].resample("MS").last()
        rise = (unemp.shift(-1) - unemp >= .2).astype(float).where(unemp.shift(-1).notna())
        add_forecast(forecasts, "unemployment_rises", "Unemployment up ≥ 0.2 pp", "jobs", "next release",
                     fit_forecast(monthly, rise, min_samples=300),
                     "Probability of a monthly rise of at least 0.2 percentage points.")

    benchmarks = []
    base12 = future_any(recession, 12).dropna().mean()
    benchmarks.append({"name": "NBER base rate", "horizon": "12 months",
                       "probability": safe_number(base12),
                       "description": "Historical recession frequency over the horizon, no explanatory variables."})
    if "term_spread" in monthly:
        spread = monthly["term_spread"].dropna()
        if not spread.empty:
            benchmarks.append({
                "name": "Treasury curve 10y–3m now", "horizon": "current",
                "value": safe_number(spread.iloc[-1]),
                "date": str(spread.index[-1].date()),
                "description": "Gap between the 10-year and 3-month Treasury; negative means inverted.",
            })
        result = fit_forecast(monthly[["term_spread"]], future_any(recession, 12), min_samples=300, overlap=12)
        benchmarks.append({"name": "Estrella–Mishkin (10y–3m curve)", "horizon": "12 months", **result})
    if "sahm_realtime" in fred and not fred["sahm_realtime"].dropna().empty:
        value = float(fred["sahm_realtime"].dropna().iloc[-1])
        benchmarks.append({"name": "Real-time Sahm rule", "horizon": "current",
                           "value": safe_number(value), "signal": bool(value >= .5),
                           "description": "Flags a recession once it reaches 0.50 pp."})

    claims_state = fred_status.get("claims", {})
    credit_state = fred_status.get("hy_spread", {})
    nfci_state = fred_status.get("nfci", {})
    indicators = []
    for args in [
        (fred, "cpi", "CPI", " index", 2), (fred, "core_cpi", "Core CPI", " index", 2),
        (fred, "unemployment", "Unemployment", "%", 2), (fred, "payroll", "Payrolls", "k", 0),
        (fred, "jolts", "JOLTS openings", "k", 0),
        (fred, "claims", claims_state.get("label", "Claims"), claims_state.get("unit", ""), 0),
        (fred, "dgs10", "10-year Treasury", "%", 2), (fred, "dgs3m", "3-month Treasury", "%", 2),
        (fred, "hy_spread", credit_state.get("label", "High yield spread"),
         credit_state.get("unit", "%"), 2),
        (fred, "nfci", nfci_state.get("label", "Financial conditions"),
         nfci_state.get("unit", ""), 2),
        (markets, "vix", "VIX", "", 2), (markets, "btc", "Bitcoin", " USD", 0),
        (markets, "sp500", "S&P 500", "", 0), (markets, "nasdaq", "Nasdaq", "", 0),
    ]:
        item = latest_indicator(*args)
        if item:
            indicators.append(item)

    ok_sources = sum(1 for x in [*market_status.values(), *fred_status.values()] if x.get("ok"))
    all_sources = len(market_status) + len(fred_status)
    warnings = []
    failed = [f"{k}: {v.get('error', 'failed')}" for k, v in {**market_status, **fred_status}.items() if not v.get("ok")]
    if failed:
        warnings.append(f"{len(failed)} series unavailable; the models run degraded instead of skipping the update.")
    warnings.append("Probabilities are statistical estimates, not investment advice.")
    warnings.append("Revised macro data is lagged one month; real-time benchmarks are flagged separately.")
    governance_summary = data_governance_summary(market_status, fred_status)
    historical_benchmarks = build_historical_benchmarks(monthly, markets)
    similar_episodes = find_similar_episodes(monthly, recession, markets)
    scenarios = build_statistical_scenarios(forecasts)
    payload = {
        "version": PREDICTOR_SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": {"sources_ok": ok_sources, "sources_total": all_sources,
                      "market_start": str(markets.index.min().date()), "market_end": str(markets.index.max().date()),
                      "macro_start": str(fred.index.min().date()) if not fred.empty else None,
                      "macro_end": str(fred.index.max().date()) if not fred.empty else None},
        "forecasts": forecasts, "benchmarks": benchmarks, "indicators": indicators,
        "historical_comparisons": historical_benchmarks,
        "similar_episodes": similar_episodes, "scenarios": scenarios,
        "sources": {"market": market_status, "macro": fred_status},
        "data_governance": governance_summary, "warnings": warnings,
        "failures": failed,
        "methodology": {
            "validation": MODEL_VALIDATION_DESCRIPTION,
            "ensemble": "Regularized logistic regression + gradient boosting.",
            "calibration": "Isotonic when the calibration window has 30+ cases of each class, Platt scaling otherwise.",
            "no_leakage": ("Macro variables shifted one month; no random splits; market targets "
                           "built on each asset's own trading sessions."),
            "revisions": ("Reference periods validated; the missing historical vintage is stated "
                          "and mitigated with a conservative lag."),
            "idempotency": "Snapshots use a stable identity of coverage, observations and probabilities.",
            "metrics": "Brier, log loss, AUC, accuracy and calibration curve, all out of sample.",
            "comparisons": "Official NBER recessions with a declared window; BTC cycles split at halving blocks.",
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-years", type=int, default=20)
    # kept so old scripts/CI calls don't break, the backtest always runs now
    parser.add_argument("--backtest", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        payload = build(max(5, args.market_years))
        append_snapshot(payload)
        titles = {item["id"]: item["title"] for item in payload["forecasts"]}
        payload["forecast_history"] = load_prediction_history(HISTORY, titles)
        atomic_json(OUTPUT, payload)
        ok = sum(1 for x in payload["forecasts"] if x.get("status") == "ok")
        print(f"ok: {ok}/{len(payload['forecasts'])} forecasts -> {OUTPUT}")
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("previous output kept")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
