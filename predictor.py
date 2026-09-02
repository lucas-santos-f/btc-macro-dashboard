"""Coleta dados públicos, treina modelos temporais e gera web/predictor.json.

O pipeline não usa LLM nem exige chave comercial. As séries FRED possuem
fallbacks diretos para BLS, BEA, Treasury, DOL, Chicago Fed e OFR; a chave
gratuita da BEA continua opcional.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
    "payroll": ("PAYEMS", "BLS", "Payroll não agrícola"),
    "unemployment": ("UNRATE", "BLS", "Desemprego"),
    "participation": ("CIVPART", "BLS", "Participação"),
    "wages": ("CES0500000003", "BLS", "Salário médio por hora"),
    "jolts": ("JTSJOL", "BLS", "Vagas JOLTS"),
    "pce_price": ("PCEPI", "BEA", "PCE"),
    "core_pce": ("PCEPILFE", "BEA", "Core PCE"),
    "real_gdp": ("GDPC1", "BEA", "PIB real"),
    "real_income": ("DSPIC96", "BEA", "Renda disponível real"),
    "consumption": ("PCE", "BEA", "Consumo pessoal"),
    "claims": ("ICSA", "Department of Labor", "Pedidos de seguro-desemprego"),
    "dgs10": ("DGS10", "U.S. Treasury", "Treasury 10 anos"),
    "dgs3m": ("DGS3MO", "U.S. Treasury", "Treasury 3 meses"),
    "real10": ("DFII10", "U.S. Treasury", "Juro real 10 anos"),
    "hy_spread": ("BAMLH0A0HYM2", "ICE/BofA", "Spread high yield"),
    "nfci": ("NFCI", "Chicago Fed", "Condições financeiras"),
    "recession": ("USREC", "NBER", "Recessão NBER"),
    "sahm_realtime": ("SAHMREALTIME", "Sahm/FRED", "Sahm Rule em tempo real"),
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
    "dgs10": ("daily_treasury_yield_curve", "BC_10YEAR", 1990, "Treasury 10 anos", "%"),
    "dgs3m": ("daily_treasury_yield_curve", "BC_3MONTH", 1990, "Treasury 3 meses", "%"),
    "real10": ("daily_treasury_real_yield_curve", "TC_10YEAR", 2003, "Juro real 10 anos", "%"),
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
        "id": "2000_2001", "nome": "2000–2001 (dot-com / recessão de 2001)",
        "inicio": "2001-03-01", "fim": "2001-11-30",
    },
    {
        "id": "2008", "nome": "Crise financeira de 2008",
        "inicio": "2007-12-01", "fim": "2009-06-30",
    },
    {
        "id": "2020", "nome": "Recessão de 2020",
        "inicio": "2020-02-01", "fim": "2020-04-30",
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


def fetch_markets(years: int) -> tuple[pd.DataFrame, dict]:
    start = (pd.Timestamp.now("UTC").tz_localize(None) - pd.DateOffset(years=years)).date()
    out, status = {}, {}
    for key, ticker in MARKETS.items():
        try:
            frame = yf.download(ticker, start=start, auto_adjust=True, progress=False,
                                threads=False, timeout=45)
            if frame.empty:
                raise RuntimeError("resposta vazia")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            series = frame["Close"].dropna().astype(float)
            series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
            out[key] = series
            status[key] = {"ok": True, "fonte": "Yahoo Finance", "ultima_data": str(series.index.max().date())}
        except Exception as exc:
            status[key] = {"ok": False, "fonte": "Yahoo Finance", "erro": str(exc)[:160]}
    return pd.DataFrame(out).sort_index(), status


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
                raise RuntimeError("resposta vazia")
            return key, series, {"ok": True, "fonte": source, "serie": series_id,
                                 "rotulo": label, "ultima_data": str(series.index.max().date())}
        except Exception as exc:
            return key, None, {"ok": False, "fonte": source, "serie": series_id,
                               "rotulo": label, "erro": str(exc)[:160]}

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
    return pd.DataFrame(out).sort_index(), status


def _apply_fallback(out: dict, status: dict, candidates, fetcher) -> None:
    """Mescla um fallback sem permitir que sua falha interrompa outros adaptadores."""
    missing = [key for key in candidates if key not in out]
    if not missing:
        return
    try:
        direct, direct_status = fetcher(missing)
    except Exception as exc:
        direct = {}
        direct_status = {key: {"ok": False, "fonte": fetcher.__name__, "erro": str(exc)[:180]}
                         for key in missing}
    for key in missing:
        series = direct.get(key)
        if series is not None and not series.dropna().empty:
            state = dict(direct_status.get(key, {}))
            state.update({"ok": True, "fallback": True, "serie_primaria": FRED[key][0]})
            out[key] = series.dropna().sort_index()
            status[key] = state
            continue
        primary = status.get(key, {"ok": False, "fonte": FRED[key][1], "serie": FRED[key][0],
                                   "rotulo": FRED[key][2]})
        fallback = direct_status.get(key, {})
        source = fallback.get("fonte", fetcher.__name__)
        primary_error = primary.get("erro", "fonte primária indisponível")
        fallback_error = fallback.get("erro", "fallback sem dados")
        status[key] = {**primary, "ok": False,
                       "erro": f"{primary_error} | {source}: {fallback_error}"[-360:],
                       "tentativas": [primary.get("fonte", "FRED"), source]}


def fetch_bls_direct(keys: list[str]) -> tuple[dict, dict]:
    """Fallback oficial BLS em blocos de 20 anos, sem chave obrigatória."""
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
                raise RuntimeError("; ".join(data.get("message") or ["BLS recusou a consulta"]))
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
            status[key] = {"ok": True, "fonte": "BLS API direta", "serie": BLS_DIRECT[key],
                           "rotulo": FRED[key][2], "ultima_data": str(series.index.max().date())}
        else:
            status[key] = {"ok": False, "fonte": "BLS API direta", "serie": BLS_DIRECT[key],
                           "rotulo": FRED[key][2], "erro": " | ".join(errors)[-300:] or "sem dados"}
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
    """Fallback NIPA oficial; usa apenas a chave gratuita quando configurada."""
    api_key = os.getenv("BEA_API_KEY", "").strip()
    out, status = {}, {}
    if not api_key:
        for key in keys:
            status[key] = {"ok": False, "fonte": "BEA API direta", "serie": BEA_DIRECT[key][0],
                           "rotulo": FRED[key][2], "erro": "BEA_API_KEY ausente"}
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
                raise RuntimeError(results["Error"].get("APIErrorDescription", "BEA recusou a consulta"))
            rows = results.get("Data") or []
            for key in group_keys:
                series = _parse_bea_rows(rows, key)
                if series.empty:
                    raise RuntimeError(f"{key}: linha NIPA não encontrada em {table}")
                out[key] = series
                status[key] = {"ok": True, "fonte": "BEA API direta", "serie": table,
                               "rotulo": FRED[key][2], "ultima_data": str(series.index.max().date())}
        except Exception as exc:
            for key in group_keys:
                if key not in out:
                    status[key] = {"ok": False, "fonte": "BEA API direta", "serie": table,
                                   "rotulo": FRED[key][2], "erro": str(exc)[:180]}
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
    """Fallback oficial para curvas nominais e reais, consultado por ano."""
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
            status[key] = {"ok": True, "fonte": "U.S. Treasury XML direto",
                           "serie": f"{dataset}:{field}", "rotulo": label, "unidade": unit,
                           "ultima_data": str(series.index.max().date()),
                           "falhas_parciais": len(errors)}
        else:
            status[key] = {"ok": False, "fonte": "U.S. Treasury XML direto",
                           "serie": f"{dataset}:{field}", "rotulo": label,
                           "erro": " | ".join(errors)[-300:] or "sem dados"}
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
            raise RuntimeError("relatório ETA 539 sem claims nacionais ajustados")
        return {"claims": series}, {"claims": {"ok": True, "fonte": "U.S. DOL ETA direto",
                "serie": "ETA 539 initial claims SA", "rotulo": FRED["claims"][2],
                "unidade": " pedidos", "ultima_data": str(series.index.max().date())}}
    except Exception as exc:
        return {}, {"claims": {"ok": False, "fonte": "U.S. DOL ETA direto",
                    "serie": "ETA 539 initial claims SA", "rotulo": FRED["claims"][2],
                    "erro": str(exc)[:180]}}


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
            raise RuntimeError("CSV NFCI sem observações")
        return {"nfci": series}, {"nfci": {"ok": True, "fonte": "Chicago Fed direto",
                "serie": "NFCI", "rotulo": FRED["nfci"][2], "unidade": " índice",
                "ultima_data": str(series.index.max().date())}}
    except Exception as exc:
        return {}, {"nfci": {"ok": False, "fonte": "Chicago Fed direto", "serie": "NFCI",
                    "rotulo": FRED["nfci"][2], "erro": str(exc)[:180]}}


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
            raise RuntimeError("CSV OFR FSI sem componente de crédito")
        return {"hy_spread": series}, {"hy_spread": {"ok": True, "fonte": "OFR FSI direto",
                "serie": "OFR FSI:Credit", "rotulo": "Estresse de crédito OFR (proxy)",
                "unidade": " índice", "proxy_para": FRED["hy_spread"][0],
                "ultima_data": str(series.index.max().date())}}
    except Exception as exc:
        return {}, {"hy_spread": {"ok": False, "fonte": "OFR FSI direto",
                    "serie": "OFR FSI:Credit", "rotulo": "Estresse de crédito OFR (proxy)",
                    "erro": str(exc)[:180]}}


def load_onchain() -> pd.DataFrame:
    path = WEB / "data.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = pd.to_datetime(data["datas"])
        wanted = {"mvrv", "fluxo_liq", "saldo_ex", "fear_greed", "hashrate"}
        columns = {s["key"]: pd.to_numeric(pd.Series(s["valores"]), errors="coerce").to_numpy()
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
    features = pd.DataFrame(index=markets.index)
    for col in markets.columns:
        s = markets[col].replace(0, np.nan)
        features[f"{col}_r1"] = np.log(s.where(s > 0)).diff()
        for window in (5, 20, 60):
            features[f"{col}_mom{window}"] = s.pct_change(window, fill_method=None)
        features[f"{col}_vol20"] = np.log(s.where(s > 0)).diff().rolling(20).std() * math.sqrt(252)
    if "sp500" in markets:
        features["sp500_drawdown"] = markets["sp500"] / markets["sp500"].cummax() - 1
    if {"t10", "t3m"} <= set(markets.columns):
        features["curve_10y_3m"] = markets["t10"] - markets["t3m"]
    if {"hyg", "tlt"} <= set(markets.columns):
        features["credit_ratio"] = markets["hyg"] / markets["tlt"]
        features["credit_mom20"] = features["credit_ratio"].pct_change(20, fill_method=None)
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
    # Uma observação macro só vira feature no mês seguinte ao mês de referência.
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
    return (series.shift(-horizon) / series > 1).astype(float).where(series.shift(-horizon).notna())


def future_drawdown(series: pd.Series, horizon: int, threshold: float) -> pd.Series:
    values = series.to_numpy(float)
    out = np.full(len(values), np.nan)
    for i in range(len(values) - horizon):
        future_min = np.nanmin(values[i + 1:i + horizon + 1])
        out[i] = float(future_min / values[i] - 1 <= threshold)
    return pd.Series(out, index=series.index)


def _models() -> tuple[Pipeline, Pipeline]:
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.35, class_weight="balanced", max_iter=3000,
                                     random_state=42)),
    ])
    boost = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=12,
                                                  learning_rate=.045, l2_regularization=2,
                                                  random_state=42)),
    ])
    return logistic, boost


def temporal_walk_forward_splits(n: int, min_train: int, max_folds: int = 5,
                                 calibration_size: int | None = None,
                                 test_size: int | None = None) -> list[tuple[int, int, int]]:
    """Cria dobras expansivas treino/calibração/teste, todas contíguas e sem sobreposição."""
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
        "detectado": bool(len(y) and (positives < 30 or rate < .10)),
        "eventos": positives,
        "nao_eventos": negatives,
        "taxa": safe_number(rate),
        "tratamento": (
            "Pesos inversos de classe no boosting, regressão logística balanceada, "
            "calibração somente em janela posterior ao treino e retração à taxa-base."
        ),
    }


def _balanced_sample_weight(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map({label: len(y) / (2 * count) for label, count in counts.items()}).to_numpy(float)


def _fit_temporal_models(train_x: pd.DataFrame, train_y: pd.Series,
                         calibration_x: pd.DataFrame, calibration_y: pd.Series):
    usable = train_x.columns[train_x.notna().any(axis=0)]
    if len(usable) == 0:
        raise ValueError("treino sem variável utilizável")
    train_x = train_x[usable]
    calibration_x = calibration_x[usable]
    if train_y.nunique() < 2 or calibration_y.nunique() < 2:
        raise ValueError("treino ou calibração temporal sem as duas classes")
    if min(train_y.value_counts()) < 4:
        raise ValueError("treino com menos de quatro exemplos em uma classe")
    logit, boost = _models()
    logit.fit(train_x, train_y)
    boost.fit(train_x, train_y, model__sample_weight=_balanced_sample_weight(train_y))
    raw_calibration = (
        logit.predict_proba(calibration_x)[:, 1]
        + boost.predict_proba(calibration_x)[:, 1]
    ) / 2
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=.01, y_max=.99)
    calibrator.fit(raw_calibration, calibration_y)
    return logit, boost, calibrator, usable


def _predict_ensemble(logit: Pipeline, boost: Pipeline, calibrator: IsotonicRegression,
                      frame: pd.DataFrame) -> np.ndarray:
    raw = (logit.predict_proba(frame)[:, 1] + boost.predict_proba(frame)[:, 1]) / 2
    return np.asarray(calibrator.predict(raw), dtype=float)


def _latest_temporal_partition(y: pd.Series, min_train: int, base_size: int) -> int | None:
    """Retorna o início da calibração mais recente que ainda contém as duas classes."""
    start = max(min_train, len(y) - base_size)
    step = max(6, base_size // 2)
    while start > min_train and y.iloc[start:].nunique() < 2:
        start = max(min_train, start - step)
    return start if start < len(y) and y.iloc[start:].nunique() == 2 else None


def fit_forecast(features: pd.DataFrame, target: pd.Series, min_samples=180) -> dict:
    all_features = features.replace([np.inf, -np.inf], np.nan).copy()
    y = target.reindex(all_features.index)
    valid = y.notna() & all_features.notna().sum(axis=1).gt(0)
    frame, y = all_features.loc[valid], y.loc[valid].astype(int)
    rare = _rare_event_summary(y)
    if len(frame) < min_samples or y.nunique() < 2 or min(rare["eventos"], rare["nao_eventos"]) < 8:
        return {
            "status": "dados_insuficientes", "amostras": int(len(frame)),
            "eventos": rare["eventos"], "eventos_raros": rare,
            "motivo": "amostra ou classe minoritária insuficiente para validação temporal",
        }

    splits = temporal_walk_forward_splits(len(frame), min_samples)
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
            "treino_inicio": str(train_x.index[0].date()),
            "treino_fim": str(train_x.index[-1].date()),
            "calibracao_inicio": str(calibration_x.index[0].date()),
            "calibracao_fim": str(calibration_x.index[-1].date()),
            "teste_inicio": str(test_x.index[0].date()),
            "teste_fim": str(test_x.index[-1].date()),
            "amostras_teste": int(len(test_x)),
            "eventos_teste": int(test_y.sum()),
        })

    if len(fold_meta) < 2 or len(set(actuals)) < 2:
        return {
            "status": "dados_insuficientes", "amostras": int(len(frame)),
            "eventos": rare["eventos"], "eventos_raros": rare,
            "motivo": "menos de duas dobras temporais válidas com ambas as classes no conjunto agregado",
            "walk_forward": {"dobras_validas": len(fold_meta), "dobras": fold_meta},
        }

    pred_oos = np.asarray(predictions, dtype=float)
    actual_oos = np.asarray(actuals, dtype=int)
    preliminary_auc = roc_auc_score(actual_oos, pred_oos)
    reliability = float(np.clip((preliminary_auc - .50) / .20, 0, 1))
    base_rate = float(y.mean())
    pred_oos = np.clip(reliability * pred_oos + (1 - reliability) * base_rate, .001, .999)

    final_calibration_size = max(12, min(90, len(frame) // 10))
    final_start = _latest_temporal_partition(y, min_samples, final_calibration_size)
    if final_start is None:
        return {
            "status": "dados_insuficientes", "amostras": int(len(frame)),
            "eventos": rare["eventos"], "eventos_raros": rare,
            "motivo": "janela temporal final sem as duas classes para calibração",
            "walk_forward": {"dobras_validas": len(fold_meta), "dobras": fold_meta},
        }
    final_train_x, final_train_y = frame.iloc[:final_start], y.iloc[:final_start]
    final_calibration_x, final_calibration_y = frame.iloc[final_start:], y.iloc[final_start:]
    try:
        logit, boost, calibrator, usable = _fit_temporal_models(
            final_train_x, final_train_y, final_calibration_x, final_calibration_y
        )
    except ValueError as exc:
        return {
            "status": "dados_insuficientes", "amostras": int(len(frame)),
            "eventos": rare["eventos"], "eventos_raros": rare, "motivo": str(exc),
            "walk_forward": {"dobras_validas": len(fold_meta), "dobras": fold_meta},
        }

    current_candidates = all_features.loc[all_features.notna().sum(axis=1).gt(0)]
    if current_candidates.empty:
        return {"status": "dados_insuficientes", "amostras": int(len(frame)),
                "eventos": rare["eventos"], "eventos_raros": rare,
                "motivo": "sem observação atual utilizável"}
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
        "acuracia": safe_number(np.mean((pred_oos >= .5) == actual_oos)),
        "taxa_base": safe_number(base_rate), "confianca_modelo": safe_number(reliability),
        "amostras_teste": int(len(actual_oos)), "amostras_total": int(len(frame)),
        "eventos": rare["eventos"], "dobras": len(fold_meta),
    }
    auc = metrics["auc"] or 0
    quality = "alta" if auc >= .70 and brier <= .20 else "moderada" if auc >= .58 else "baixa"
    bins = min(8, max(3, len(actual_oos) // 25))
    observed, predicted = calibration_curve(actual_oos, pred_oos, n_bins=bins, strategy="quantile")
    calibration = [{"previsto": safe_number(p), "observado": safe_number(o)}
                   for p, o in zip(predicted, observed)]
    error = min(.35, max(.05, 1.96 * math.sqrt(max(brier, .001) / max(len(actual_oos), 1))))

    names = list(usable)
    imputed = logit.named_steps["imputer"].transform(current_x)
    scaled = logit.named_steps["scale"].transform(imputed)[0]
    coef = logit.named_steps["model"].coef_[0]
    contributions = coef[:len(names)] * scaled[:len(names)]
    order = np.argsort(np.abs(contributions))[::-1][:8]
    drivers = [{"nome": names[i], "impacto": safe_number(contributions[i]),
                "direcao": "aumenta" if contributions[i] >= 0 else "reduz"} for i in order]
    history = [
        {"data": str(pd.Timestamp(date).date()), "probabilidade": safe_number(prob), "ocorreu": int(actual)}
        for date, prob, actual in zip(dates[-240:], pred_oos[-240:], actual_oos[-240:])
    ]
    return {
        "status": "ok", "probabilidade": safe_number(probability), "qualidade": quality,
        "intervalo": [safe_number(max(0, probability - error)),
                      safe_number(min(1, probability + error))],
        "metricas": metrics, "calibracao": calibration, "drivers": drivers,
        "historico_teste": history,
        "ultima_observacao": str(current_candidates.index[-1].date()),
        "eventos_raros": rare,
        "walk_forward": {
            "metodo": "janela expansiva; calibração e teste contíguos posteriores ao treino",
            "dobras_validas": len(fold_meta), "dobras": fold_meta,
            "previsoes_fora_amostra": int(len(actual_oos)),
        },
        "calibracao_temporal": {
            "metodo": "isotônica",
            "treino_fim": str(final_train_x.index[-1].date()),
            "calibracao_inicio": str(final_calibration_x.index[0].date()),
            "calibracao_fim": str(final_calibration_x.index[-1].date()),
        },
    }


def add_forecast(collection: list, forecast_id: str, title: str, kind: str,
                 horizon: str, result: dict, description: str) -> None:
    collection.append({"id": forecast_id, "titulo": title, "tipo": kind,
                       "horizonte": horizon, "descricao": description, **result})


def latest_indicator(frame: pd.DataFrame, key: str, label: str, unit="", digits=2) -> dict | None:
    if key not in frame or frame[key].dropna().empty:
        return None
    series = frame[key].dropna()
    current = series.iloc[-1]
    previous = series.iloc[-2] if len(series) > 1 else np.nan
    return {"id": key, "nome": label, "valor": safe_number(current, digits), "unidade": unit,
            "variacao": safe_number(current - previous, digits), "data": str(series.index[-1].date())}



def period_metrics(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    """Resume apenas observações existentes na janela; não interpola dados ausentes."""
    clean = series.dropna().sort_index()
    window = clean.loc[(clean.index >= start) & (clean.index <= end)]
    if len(window) < 2:
        return None
    returns = window.pct_change(fill_method=None).dropna()
    drawdown = window / window.cummax() - 1
    return {
        "inicio_observado": str(window.index[0].date()),
        "fim_observado": str(window.index[-1].date()),
        "observacoes": int(len(window)),
        "retorno": safe_number(window.iloc[-1] / window.iloc[0] - 1),
        "drawdown_maximo": safe_number(drawdown.min()),
        "volatilidade_anualizada": safe_number(returns.std() * math.sqrt(252)) if len(returns) > 1 else None,
    }


def recession_comparisons(monthly: pd.DataFrame, markets: pd.DataFrame) -> list[dict]:
    """Compara janelas fixas ao redor das recessões oficiais, deixando lacunas explícitas."""
    output = []
    for episode in NBER_EPISODES:
        recession_start = pd.Timestamp(episode["inicio"])
        recession_end = pd.Timestamp(episode["fim"])
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
                    "valor_pre_recessao": safe_number(before.iloc[-1]) if not before.empty else None,
                    "minimo_janela": safe_number(curve_window.min()),
                    "maximo_janela": safe_number(curve_window.max()),
                    "inicio_observado": str(curve_window.index[0].date()),
                    "fim_observado": str(curve_window.index[-1].date()),
                }
        output.append({
            "id": episode["id"], "nome": episode["nome"],
            "periodo_nber": [str(recession_start.date()), str(recession_end.date())],
            "janela_analitica": [str(analysis_start.date()), str(analysis_end.date())],
            "regra_janela": "12 meses antes do início NBER até 6 meses após o fim NBER",
            "ativos": assets, "curva_10a_3m": curve,
            "cobertura": "completa" if not missing else "parcial",
            "series_ausentes": missing,
            "origem": {
                "recessoes": "NBER Business Cycle Dating Committee",
                "url": NBER_SOURCE_URL,
                "mercados": "Yahoo Finance; somente fechamentos existentes na cobertura coletada",
            },
        })
    return output


def btc_cycle_comparisons(markets: pd.DataFrame) -> list[dict]:
    """Segmenta os ciclos disponíveis por halvings verificáveis na blockchain."""
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
        observed_start = pd.Timestamp(metrics["inicio_observado"])
        complete_start = abs((observed_start - start).days) <= 7
        output.append({
            "id": f"halving_{block}", "nome": f"Ciclo após halving do bloco {block}",
            "inicio": str(start.date()), "fim": str(end.date()),
            "completo_desde_halving": complete_start,
            "metricas": metrics,
            "origem": "Bitcoin blockchain: timestamp público do bloco de halving; preços Yahoo Finance",
            "limitacao": (
                None if complete_start
                else "A série de preços disponível começa depois do halving; métricas cobrem só o trecho observado."
            ),
        })
    return output


def build_historical_benchmarks(monthly: pd.DataFrame, markets: pd.DataFrame) -> dict:
    return {
        "recessoes": recession_comparisons(monthly, markets),
        "ciclos_btc": btc_cycle_comparisons(markets),
        "limitacoes": [
            "Comparações são descritivas e não implicam repetição do episódio.",
            "Valores ausentes permanecem nulos; não há interpolação nem reconstrução de preços.",
            "Datas de recessão seguem o NBER; janelas de mercado são regras analíticas declaradas.",
        ],
    }


def append_snapshot(payload: dict) -> None:
    row = {"gerado_em": payload["gerado_em"]}
    for item in payload["previsoes"]:
        row[item["id"]] = item.get("probabilidade")
    new = pd.DataFrame([row])
    if HISTORY.exists():
        old = pd.read_csv(HISTORY)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates("gerado_em", keep="last")
    new.to_csv(HISTORY, index=False)


def build(years: int) -> dict:
    markets, market_status = fetch_markets(years)
    fred, fred_status = fetch_fred()
    if markets.empty:
        raise RuntimeError("Nenhuma série de mercado respondeu; saída anterior preservada.")
    onchain = load_onchain()
    daily = daily_features(markets, onchain)
    monthly, recession = monthly_features(fred, markets)
    forecasts: list[dict] = []

    for horizon in (6, 12):
        result = fit_forecast(monthly, future_any(recession, horizon), min_samples=180)
        add_forecast(forecasts, f"recessao_{horizon}m", "Recessão dos EUA", "macro",
                     f"{horizon} meses", result,
                     "Probabilidade de existir ao menos um mês NBER de recessão no horizonte.")

    if "sp500" in markets:
        for days, threshold in ((30, -.10), (90, -.10), (180, -.20)):
            result = fit_forecast(daily, future_drawdown(markets["sp500"], days, threshold), min_samples=700)
            add_forecast(forecasts, f"sp_stress_{days}d", f"S&P 500: queda de {abs(threshold):.0%}",
                         "estresse", f"{days} pregões", result,
                         "Probabilidade de o menor fechamento futuro ultrapassar o limite de queda.")

    for asset, label in (("btc", "Bitcoin"), ("nasdaq", "Nasdaq"), ("sp500", "S&P 500")):
        if asset not in markets:
            continue
        for horizon in (1, 5, 20, 60):
            result = fit_forecast(daily, future_direction(markets[asset], horizon), min_samples=700)
            add_forecast(forecasts, f"{asset}_alta_{horizon}d", f"{label}: retorno positivo",
                         "direcao", f"{horizon} pregões", result,
                         "Probabilidade calibrada de o fechamento futuro superar o atual.")

    # Eventos macro mensais. A comparação é com o valor anterior, não consenso de mercado.
    if "cpi" in fred:
        cpi_mom = fred["cpi"].resample("MS").last().pct_change(fill_method=None)
        target = (cpi_mom.shift(-1) > cpi_mom).astype(float).where(cpi_mom.shift(-1).notna())
        add_forecast(forecasts, "cpi_acelera", "CPI acelera", "inflacao", "próxima divulgação",
                     fit_forecast(monthly, target, min_samples=300),
                     "Probabilidade de a variação mensal superar a leitura mensal anterior.")
    if "core_pce" in fred:
        pce_mom = fred["core_pce"].resample("MS").last().pct_change(fill_method=None)
        target = (pce_mom.shift(-1) > pce_mom).astype(float).where(pce_mom.shift(-1).notna())
        add_forecast(forecasts, "pce_acelera", "Core PCE acelera", "inflacao", "próxima divulgação",
                     fit_forecast(monthly, target, min_samples=300),
                     "Probabilidade de a variação mensal superar a leitura mensal anterior.")
    if "payroll" in fred:
        payroll_change = fred["payroll"].resample("MS").last().diff()
        slower = (payroll_change.shift(-1) < payroll_change).astype(float).where(payroll_change.shift(-1).notna())
        negative = (payroll_change.shift(-1) < 0).astype(float).where(payroll_change.shift(-1).notna())
        add_forecast(forecasts, "payroll_desacelera", "Payroll desacelera", "emprego", "próxima divulgação",
                     fit_forecast(monthly, slower, min_samples=300),
                     "Probabilidade de a criação líquida de vagas ficar abaixo da leitura anterior.")
        add_forecast(forecasts, "payroll_negativo", "Payroll negativo", "emprego", "próxima divulgação",
                     fit_forecast(monthly, negative, min_samples=300),
                     "Probabilidade de destruição líquida de vagas; evento historicamente raro.")
    if "unemployment" in fred:
        unemp = fred["unemployment"].resample("MS").last()
        rise = (unemp.shift(-1) - unemp >= .2).astype(float).where(unemp.shift(-1).notna())
        add_forecast(forecasts, "desemprego_sobe", "Desemprego sobe ≥ 0,2 pp", "emprego", "próxima divulgação",
                     fit_forecast(monthly, rise, min_samples=300),
                     "Probabilidade de aumento mensal de pelo menos 0,2 ponto percentual.")

    benchmarks = []
    base12 = future_any(recession, 12).dropna().mean()
    benchmarks.append({"nome": "Frequência-base NBER", "horizonte": "12 meses",
                       "probabilidade": safe_number(base12),
                       "descricao": "Frequência histórica de recessão no horizonte, sem variáveis explicativas."})
    if "term_spread" in monthly:
        spread = monthly["term_spread"].dropna()
        if not spread.empty:
            benchmarks.append({
                "nome": "Curva Treasury 10a–3m atual", "horizonte": "estado atual",
                "valor": safe_number(spread.iloc[-1]),
                "data": str(spread.index[-1].date()),
                "descricao": "Diferença observada entre Treasury de 10 anos e 3 meses; valor negativo indica inversão.",
            })
        result = fit_forecast(monthly[["term_spread"]], future_any(recession, 12), min_samples=300)
        benchmarks.append({"nome": "Estrella–Mishkin (curva 10a–3m)", "horizonte": "12 meses", **result})
    if "sahm_realtime" in fred and not fred["sahm_realtime"].dropna().empty:
        value = float(fred["sahm_realtime"].dropna().iloc[-1])
        benchmarks.append({"nome": "Sahm Rule em tempo real", "horizonte": "estado atual",
                           "valor": safe_number(value), "sinal": bool(value >= .5),
                           "descricao": "Sinaliza recessão quando alcança 0,50 pp."})

    claims_state = fred_status.get("claims", {})
    credit_state = fred_status.get("hy_spread", {})
    nfci_state = fred_status.get("nfci", {})
    indicators = []
    for args in [
        (fred, "cpi", "CPI", " índice", 2), (fred, "core_cpi", "Core CPI", " índice", 2),
        (fred, "unemployment", "Desemprego", "%", 2), (fred, "payroll", "Payroll", " mil", 0),
        (fred, "jolts", "Vagas JOLTS", " mil", 0),
        (fred, "claims", claims_state.get("rotulo", "Claims"), claims_state.get("unidade", ""), 0),
        (fred, "dgs10", "Treasury 10 anos", "%", 2), (fred, "dgs3m", "Treasury 3 meses", "%", 2),
        (fred, "hy_spread", credit_state.get("rotulo", "Spread high yield"),
         credit_state.get("unidade", "%"), 2),
        (fred, "nfci", nfci_state.get("rotulo", "Condições financeiras"),
         nfci_state.get("unidade", ""), 2),
        (markets, "vix", "VIX", "", 2), (markets, "btc", "Bitcoin", " US$", 0),
        (markets, "sp500", "S&P 500", "", 0), (markets, "nasdaq", "Nasdaq", "", 0),
    ]:
        item = latest_indicator(*args)
        if item:
            indicators.append(item)

    ok_sources = sum(1 for x in [*market_status.values(), *fred_status.values()] if x.get("ok"))
    all_sources = len(market_status) + len(fred_status)
    warnings = []
    failed = [f"{k}: {v.get('erro', 'falhou')}" for k, v in {**market_status, **fred_status}.items() if not v.get("ok")]
    if failed:
        warnings.append(f"{len(failed)} séries indisponíveis; modelos degradados sem interromper a atualização.")
    warnings.append("Probabilidades são estimativas estatísticas, não recomendação de investimento.")
    warnings.append("Dados macro revisados são defasados em um mês; benchmarks em tempo real são identificados separadamente.")
    historical_benchmarks = build_historical_benchmarks(monthly, markets)
    payload = {
        "versao": 2, "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cobertura": {"fontes_ok": ok_sources, "fontes_total": all_sources,
                      "mercado_inicio": str(markets.index.min().date()), "mercado_fim": str(markets.index.max().date()),
                      "macro_inicio": str(fred.index.min().date()) if not fred.empty else None,
                      "macro_fim": str(fred.index.max().date()) if not fred.empty else None},
        "previsoes": forecasts, "benchmarks": benchmarks, "indicadores": indicators,
        "comparacoes_historicas": historical_benchmarks,
        "fontes": {"mercado": market_status, "macro": fred_status}, "avisos": warnings,
        "falhas": failed,
        "metodologia": {
            "validacao": "Divisão cronológica 55% treino, 23% calibração isotônica e 22% teste.",
            "ensemble": "Regressão logística regularizada + gradient boosting.",
            "anti_vazamento": "Variáveis macro deslocadas um mês; nenhuma divisão aleatória.",
            "metricas": "Brier, log loss, AUC, acurácia e curva de calibração fora da amostra.",
            "comparacoes": "Recessões oficiais NBER com janela declarada; ciclos BTC segmentados por blocos de halving.",
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anos-mercado", type=int, default=20)
    parser.add_argument("--backtest", action="store_true", help="Mantido para interface; o backtest já faz parte da execução.")
    args = parser.parse_args()
    try:
        payload = build(max(5, args.anos_mercado))
        atomic_json(OUTPUT, payload)
        append_snapshot(payload)
        ok = sum(1 for x in payload["previsoes"] if x.get("status") == "ok")
        print(f"OK: {ok}/{len(payload['previsoes'])} previsões válidas -> {OUTPUT}")
        return 0
    except Exception as exc:
        print(f"FALHA: {type(exc).__name__}: {exc}")
        print("A saída anterior foi preservada.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
