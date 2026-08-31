"""Coleta dados públicos, treina modelos temporais e gera web/predictor.json.

O pipeline não usa LLM nem exige chave. FRED, BLS, BEA, Treasury e NBER são
representados por séries públicas do FRED Graph CSV; chaves opcionais podem ser
fornecidas por variáveis de ambiente sem entrar no repositório.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import tempfile
from datetime import datetime, timezone
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

RECESSIONS = [
    ("1960-05", "1961-02"), ("1969-12", "1970-11"),
    ("1973-11", "1975-03"), ("1980-01", "1980-07"),
    ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
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
            from io import StringIO
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
    missing_bls = [key for key in BLS_DIRECT if key not in out]
    if missing_bls:
        direct, direct_status = fetch_bls_direct(missing_bls)
        out.update(direct)
        status.update(direct_status)
    return pd.DataFrame(out).sort_index(), status


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


def fit_forecast(features: pd.DataFrame, target: pd.Series, min_samples=180) -> dict:
    frame = features.copy()
    y = target.reindex(frame.index)
    valid = y.notna() & frame.notna().sum(axis=1).gt(0)
    frame, y = frame.loc[valid], y.loc[valid].astype(int)
    if len(frame) < min_samples or y.nunique() < 2 or y.sum() < 8:
        return {"status": "dados_insuficientes", "amostras": int(len(frame)),
                "eventos": int(y.sum()) if len(y) else 0}
    n = len(frame)
    a, b = max(int(n * .55), 60), max(int(n * .78), 90)
    if b >= n - 15:
        b = n - 15
    train_x, train_y = frame.iloc[:a], y.iloc[:a]
    cal_x, cal_y = frame.iloc[a:b], y.iloc[a:b]
    test_x, test_y = frame.iloc[b:], y.iloc[b:]
    usable = train_x.columns[train_x.notna().any(axis=0)]
    train_x, cal_x, test_x = train_x[usable], cal_x[usable], test_x[usable]
    frame = frame[usable]
    if min(train_y.nunique(), cal_y.nunique(), test_y.nunique()) < 2:
        return {"status": "dados_insuficientes", "amostras": int(n), "eventos": int(y.sum()),
                "motivo": "divisão temporal sem as duas classes"}
    logit, boost = _models()
    logit.fit(train_x, train_y)
    boost.fit(train_x, train_y)
    raw_cal = (logit.predict_proba(cal_x)[:, 1] + boost.predict_proba(cal_x)[:, 1]) / 2
    # Limites conservadores evitam probabilidades de 0%/100% em amostras finitas.
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=.02, y_max=.98)
    calibrator.fit(raw_cal, cal_y)
    raw_test = (logit.predict_proba(test_x)[:, 1] + boost.predict_proba(test_x)[:, 1]) / 2
    pred_test = calibrator.predict(raw_test)
    current_x = frame.iloc[[-1]]
    raw_current = (logit.predict_proba(current_x)[:, 1] + boost.predict_proba(current_x)[:, 1]) / 2
    probability = float(calibrator.predict(raw_current)[0])
    # Evita falsa precisão: modelos sem poder discriminatório retornam à taxa-base.
    preliminary_auc = roc_auc_score(test_y, pred_test)
    reliability = float(np.clip((preliminary_auc - .50) / .20, 0, 1))
    base_rate = float(pd.concat([train_y, cal_y]).mean())
    pred_test = reliability * pred_test + (1 - reliability) * base_rate
    probability = reliability * probability + (1 - reliability) * base_rate
    brier = brier_score_loss(test_y, pred_test)
    metrics = {
        "brier": safe_number(brier),
        "log_loss": safe_number(log_loss(test_y, pred_test, labels=[0, 1])),
        "auc": safe_number(roc_auc_score(test_y, pred_test)),
        "acuracia": safe_number(np.mean((pred_test >= .5) == test_y.to_numpy())),
        "taxa_base": safe_number(base_rate), "confianca_modelo": safe_number(reliability),
        "amostras_teste": int(len(test_y)), "amostras_total": int(n), "eventos": int(y.sum()),
    }
    auc = metrics["auc"] or 0
    quality = "alta" if auc >= .70 and brier <= .20 else "moderada" if auc >= .58 else "baixa"
    observed, predicted = calibration_curve(test_y, pred_test, n_bins=min(8, max(3, len(test_y) // 25)), strategy="quantile")
    calibration = [{"previsto": safe_number(p), "observado": safe_number(o)}
                   for p, o in zip(predicted, observed)]
    error = min(.35, max(.05, 1.96 * math.sqrt(max(brier, .001) / max(len(test_y), 1))))

    # Contribuições do modelo linear na escala padronizada.
    names = list(frame.columns)
    imputed = logit.named_steps["imputer"].transform(current_x)
    scaled = logit.named_steps["scale"].transform(imputed)[0]
    coef = logit.named_steps["model"].coef_[0]
    contributions = coef[:len(names)] * scaled[:len(names)]
    order = np.argsort(np.abs(contributions))[::-1][:8]
    drivers = [{"nome": names[i], "impacto": safe_number(contributions[i]),
                "direcao": "aumenta" if contributions[i] >= 0 else "reduz"} for i in order]
    history = [{"data": str(date.date()), "probabilidade": safe_number(prob), "ocorreu": int(actual)}
               for date, prob, actual in zip(test_x.index[-180:], pred_test[-180:], test_y.to_numpy()[-180:])]
    return {
        "status": "ok", "probabilidade": safe_number(probability),
        "qualidade": quality,
        "intervalo": [safe_number(max(0, probability - error)), safe_number(min(1, probability + error))],
        "metricas": metrics, "calibracao": calibration, "drivers": drivers,
        "historico_teste": history, "ultima_observacao": str(frame.index[-1].date()),
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
        result = fit_forecast(monthly[["term_spread"]], future_any(recession, 12), min_samples=300)
        benchmarks.append({"nome": "Estrella–Mishkin (curva 10a–3m)", "horizonte": "12 meses", **result})
    if "sahm_realtime" in fred and not fred["sahm_realtime"].dropna().empty:
        value = float(fred["sahm_realtime"].dropna().iloc[-1])
        benchmarks.append({"nome": "Sahm Rule em tempo real", "horizonte": "estado atual",
                           "valor": safe_number(value), "sinal": bool(value >= .5),
                           "descricao": "Sinaliza recessão quando alcança 0,50 pp."})

    indicators = []
    for args in [
        (fred, "cpi", "CPI", " índice", 2), (fred, "core_cpi", "Core CPI", " índice", 2),
        (fred, "unemployment", "Desemprego", "%", 2), (fred, "payroll", "Payroll", " mil", 0),
        (fred, "jolts", "Vagas JOLTS", " mil", 0), (fred, "claims", "Claims", "", 0),
        (fred, "dgs10", "Treasury 10 anos", "%", 2), (fred, "dgs3m", "Treasury 3 meses", "%", 2),
        (fred, "hy_spread", "Spread high yield", "%", 2), (fred, "nfci", "Condições financeiras", "", 2),
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
    payload = {
        "versao": 1, "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cobertura": {"fontes_ok": ok_sources, "fontes_total": all_sources,
                      "mercado_inicio": str(markets.index.min().date()), "mercado_fim": str(markets.index.max().date()),
                      "macro_inicio": str(fred.index.min().date()) if not fred.empty else None,
                      "macro_fim": str(fred.index.max().date()) if not fred.empty else None},
        "previsoes": forecasts, "benchmarks": benchmarks, "indicadores": indicators,
        "fontes": {"mercado": market_status, "macro": fred_status}, "avisos": warnings,
        "falhas": failed,
        "metodologia": {
            "validacao": "Divisão cronológica 55% treino, 23% calibração isotônica e 22% teste.",
            "ensemble": "Regressão logística regularizada + gradient boosting.",
            "anti_vazamento": "Variáveis macro deslocadas um mês; nenhuma divisão aleatória.",
            "metricas": "Brier, log loss, AUC, acurácia e curva de calibração fora da amostra.",
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
