"""
Registry de series e adaptadores de fonte.

Para adicionar uma metrica nova, acrescente uma entrada em SERIES. Se a fonte
ainda nao existir, escreva um adaptador `busca_<fonte>()` que devolva um
DataFrame indexado por data e registre-o em ADAPTADORES.

Campos de cada serie:
  key      identificador interno (vira chave no data.json)
  label    nome exibido
  grupo    id do grupo (ver GRUPOS)
  eixo     "indexado" -> painel principal, normalizado base 100 em escala log
           "nativo"   -> painel inferior, escala propria (o nivel absoluto
                         tem significado: MVRV=1, Fear&Greed=50, juros=4%)
  fonte    chave em ADAPTADORES
  id       identificador da serie na fonte
  cor / prefixo / sufixo / casas  formatacao
  padrao   True = marcada ao abrir o dashboard
"""

import io
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (dashboard local)"}
TIMEOUT = 90

GRUPOS = [
    {"id": "btc",       "nome": "Bitcoin"},
    {"id": "valuation", "nome": "Valuation on-chain"},
    {"id": "rede",      "nome": "Rede"},
    {"id": "exchanges", "nome": "Exchanges"},
    {"id": "sentimento","nome": "Sentimento"},
    {"id": "indices",   "nome": "Índices"},
    {"id": "juros",     "nome": "Juros e volatilidade"},
    {"id": "commod",    "nome": "Commodities e crédito"},
]

# ---------------------------------------------------------------------------

SERIES = [
    # --- Bitcoin -----------------------------------------------------------
    dict(key="btc", label="Bitcoin", grupo="btc", eixo="indexado", padrao=True,
         fonte="coinmetrics", id="PriceUSD", cor="#F7931A", prefixo="$", casas=0),
    dict(key="realized_price", label="Realized Price", grupo="valuation", eixo="indexado", padrao=True,
         fonte="derivada", id="realized_price", cor="#FFD166", prefixo="$", casas=0),
    dict(key="mktcap", label="Market Cap", grupo="valuation", eixo="indexado",
         fonte="coinmetrics", id="CapMrktCurUSD", cor="#E8A33D", prefixo="$", casas=0),

    # --- Valuation (osciladores) -------------------------------------------
    # MVRV e, por definicao, Preco/RealizedPrice — nao criar uma serie separada
    # para essa razao, seria a mesma linha com outro nome.
    dict(key="mvrv", label="MVRV", grupo="valuation", eixo="nativo", padrao=True,
         fonte="coinmetrics", id="CapMVRVCur", cor="#F472B6", casas=2),
    dict(key="roi30d", label="ROI 30 dias", grupo="valuation", eixo="nativo",
         fonte="coinmetrics", id="ROI30d", cor="#A78BFA", sufixo="%", casas=1, escala=100),

    # --- Rede --------------------------------------------------------------
    dict(key="hashrate", label="Hash Rate", grupo="rede", eixo="indexado",
         fonte="coinmetrics", id="HashRate", cor="#34D399", sufixo=" TH/s", casas=0),
    dict(key="adr_ativos", label="Endereços ativos", grupo="rede", eixo="indexado",
         fonte="coinmetrics", id="AdrActCnt", cor="#2DD4BF", casas=0),
    dict(key="adr_saldo", label="Endereços com saldo", grupo="rede", eixo="indexado",
         fonte="coinmetrics", id="AdrBalCnt", cor="#5EEAD4", casas=0),
    dict(key="txs", label="Transações", grupo="rede", eixo="indexado",
         fonte="coinmetrics", id="TxCnt", cor="#22D3EE", casas=0),
    dict(key="emissao_usd", label="Emissão diária", grupo="rede", eixo="indexado",
         fonte="coinmetrics", id="IssTotUSD", cor="#0EA5E9", prefixo="$", casas=0),
    dict(key="taxas_btc", label="Taxas pagas", grupo="rede", eixo="indexado",
         fonte="coinmetrics", id="FeeTotNtv", cor="#38BDF8", sufixo=" BTC", casas=2),
    dict(key="dificuldade", label="Dificuldade", grupo="rede", eixo="indexado",
         fonte="blockchain", id="difficulty", cor="#4ADE80", casas=0),
    dict(key="utxos", label="UTXOs", grupo="rede", eixo="indexado",
         fonte="blockchain", id="utxo-count", cor="#86EFAC", casas=0),
    dict(key="rev_mineradores", label="Receita mineradores", grupo="rede", eixo="indexado",
         fonte="blockchain", id="miners-revenue", cor="#16A34A", prefixo="$", casas=0),

    # --- Exchanges ---------------------------------------------------------
    dict(key="saldo_ex", label="Saldo em exchanges", grupo="exchanges", eixo="indexado", padrao=True,
         fonte="coinmetrics", id="SplyExNtv", cor="#F87171", sufixo=" BTC", casas=0),
    dict(key="fluxo_ent", label="Entrada em exchanges", grupo="exchanges", eixo="indexado",
         fonte="coinmetrics", id="FlowInExUSD", cor="#FCA5A5", prefixo="$", casas=0),
    dict(key="fluxo_sai", label="Saída de exchanges", grupo="exchanges", eixo="indexado",
         fonte="coinmetrics", id="FlowOutExUSD", cor="#FDBA74", prefixo="$", casas=0),
    dict(key="fluxo_liq", label="Fluxo líquido (ent−saí)", grupo="exchanges", eixo="nativo",
         fonte="derivada", id="fluxo_liq", cor="#EF4444", prefixo="$", casas=0),

    # --- Sentimento --------------------------------------------------------
    dict(key="fear_greed", label="Fear & Greed", grupo="sentimento", eixo="nativo", padrao=True,
         fonte="feargreed", id="fng", cor="#FACC15", casas=0),

    # --- Indices -----------------------------------------------------------
    dict(key="dxy", label="DXY", grupo="indices", eixo="indexado", padrao=True,
         fonte="yahoo", id="DX-Y.NYB", cor="#4ADE80", casas=2),
    dict(key="nasdaq", label="Nasdaq", grupo="indices", eixo="indexado", padrao=True,
         fonte="yahoo", id="^IXIC", cor="#60A5FA", casas=0),
    dict(key="sp500", label="S&P 500", grupo="indices", eixo="indexado", padrao=True,
         fonte="yahoo", id="^GSPC", cor="#C084FC", casas=0),
    dict(key="russell", label="Russell 2000", grupo="indices", eixo="indexado",
         fonte="yahoo", id="^RUT", cor="#818CF8", casas=0),

    # --- Juros e volatilidade (osciladores) --------------------------------
    dict(key="t10", label="Treasury 10 anos", grupo="juros", eixo="nativo",
         fonte="yahoo", id="^TNX", cor="#94A3B8", sufixo="%", casas=2),
    dict(key="t30", label="Treasury 30 anos", grupo="juros", eixo="nativo",
         fonte="yahoo", id="^TYX", cor="#CBD5E1", sufixo="%", casas=2),
    dict(key="t13s", label="T-bill 13 semanas", grupo="juros", eixo="nativo",
         fonte="yahoo", id="^IRX", cor="#64748B", sufixo="%", casas=2),
    dict(key="curva", label="Curva 10a − 13sem", grupo="juros", eixo="nativo",
         fonte="derivada", id="curva", cor="#E2E8F0", sufixo="pp", casas=2),
    dict(key="vix", label="VIX", grupo="juros", eixo="nativo",
         fonte="yahoo", id="^VIX", cor="#FB923C", casas=1),

    # --- Commodities e credito ---------------------------------------------
    dict(key="ouro", label="Ouro", grupo="commod", eixo="indexado",
         fonte="yahoo", id="GC=F", cor="#EAB308", prefixo="$", casas=0),
    dict(key="prata", label="Prata", grupo="commod", eixo="indexado",
         fonte="yahoo", id="SI=F", cor="#D4D4D8", prefixo="$", casas=2),
    dict(key="cobre", label="Cobre", grupo="commod", eixo="indexado",
         fonte="yahoo", id="HG=F", cor="#EA580C", prefixo="$", casas=2),
    dict(key="petroleo", label="Petróleo WTI", grupo="commod", eixo="indexado",
         fonte="yahoo", id="CL=F", cor="#78716C", prefixo="$", casas=2),
    dict(key="tlt", label="Bonds 20a+ (TLT)", grupo="commod", eixo="indexado",
         fonte="yahoo", id="TLT", cor="#A8A29E", prefixo="$", casas=2),
    dict(key="hyg", label="High Yield (HYG)", grupo="commod", eixo="indexado",
         fonte="yahoo", id="HYG", cor="#D6D3D1", prefixo="$", casas=2),
]


# --------------------------------------------------------------------------
# Adaptadores. Cada um recebe (ids, inicio, fim) e devolve DataFrame
# indexado por data (DatetimeIndex) com uma coluna por id.
# --------------------------------------------------------------------------

def busca_yahoo(ids, inicio, fim):
    import yfinance as yf
    out = {}
    for tk in ids:
        df = yf.download(tk, start=inicio, end=fim, interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            print(f"    ! {tk}: vazio")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out[tk] = df["Close"].dropna()
    return pd.DataFrame(out)


def busca_coinmetrics(ids, inicio, fim):
    """Community API: sem chave, 31 metricas liberadas para BTC."""
    r = requests.get(
        "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
        params={"assets": "btc", "metrics": ",".join(ids), "frequency": "1d",
                "start_time": inicio.strftime("%Y-%m-%d"),
                "end_time": fim.strftime("%Y-%m-%d"), "page_size": 10000},
        headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    linhas = r.json()["data"]
    df = pd.DataFrame(linhas)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("time").drop(columns=["asset"], errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce")


def busca_feargreed(ids, inicio, fim):
    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0},
                     headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()["data"]
    s = pd.Series(
        [float(x["value"]) for x in d],
        index=pd.to_datetime([int(x["timestamp"]) for x in d], unit="s").normalize(),
    ).sort_index()
    return pd.DataFrame({"fng": s})


def busca_blockchain(ids, inicio, fim):
    """api.blockchain.info/charts — sem chave. Reamostra para diario."""
    out = {}
    for nome in ids:
        for tentativa in range(3):
            try:
                r = requests.get(f"https://api.blockchain.info/charts/{nome}",
                                 params={"timespan": "8years", "format": "json"},
                                 headers=UA, timeout=TIMEOUT)
                r.raise_for_status()
                v = r.json().get("values") or []
                if not v:
                    raise RuntimeError("sem values")
                s = pd.Series([p["y"] for p in v],
                              index=pd.to_datetime([p["x"] for p in v], unit="s"))
                # algumas series vem em alta frequencia; ultimo ponto de cada dia
                out[nome] = s.resample("D").last().dropna()
                break
            except Exception as e:
                if tentativa == 2:
                    print(f"    ! {nome}: {type(e).__name__} — pulando")
                else:
                    time.sleep(2)
    return pd.DataFrame(out)


ADAPTADORES = {
    "yahoo": busca_yahoo,
    "coinmetrics": busca_coinmetrics,
    "feargreed": busca_feargreed,
    "blockchain": busca_blockchain,
}


def calcula_derivadas(bruto: pd.DataFrame) -> dict:
    """Metricas que nao vem prontas de nenhuma fonte."""
    d = {}
    if {"PriceUSD", "CapMVRVCur"} <= set(bruto.columns):
        # MVRV = MktCap/RealizedCap = Preco/RealizedPrice  =>  RP = Preco/MVRV
        d["realized_price"] = bruto["PriceUSD"] / bruto["CapMVRVCur"]
    if {"FlowInExUSD", "FlowOutExUSD"} <= set(bruto.columns):
        d["fluxo_liq"] = bruto["FlowInExUSD"] - bruto["FlowOutExUSD"]
    if {"^TNX", "^IRX"} <= set(bruto.columns):
        d["curva"] = bruto["^TNX"] - bruto["^IRX"]
    return d
