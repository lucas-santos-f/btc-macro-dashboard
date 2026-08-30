"""
Baixa todas as series registradas em fontes.py e grava web/data.json.

Uso:
    python fetch_data.py            # ultimos 7 anos
    python fetch_data.py --anos 10

O dashboard le apenas o data.json gerado aqui: o navegador nao faz nenhuma
chamada de rede. Este script e a unica saida de rede do projeto.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fontes import ADAPTADORES, GRUPOS, SERIES, calcula_derivadas

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "web" / "data.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anos", type=int, default=7)
    args = ap.parse_args()

    fim = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    inicio = fim - pd.DateOffset(years=args.anos)
    print(f"Janela: {inicio.date()} -> {fim.date()}\n")

    # agrupa por fonte para minimizar requisicoes
    por_fonte = defaultdict(list)
    for s in SERIES:
        if s["fonte"] != "derivada":
            por_fonte[s["fonte"]].append(s["id"])

    bruto = pd.DataFrame()
    for fonte, ids in por_fonte.items():
        ids = list(dict.fromkeys(ids))
        print(f"  {fonte} ({len(ids)})...", flush=True)
        try:
            df = ADAPTADORES[fonte](ids, inicio, fim)
        except Exception as e:
            print(f"    FALHOU: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            print("    vazio", file=sys.stderr)
            continue
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")]
        bruto = df if bruto.empty else bruto.join(df, how="outer")

    if bruto.empty:
        print("Nenhuma fonte respondeu.", file=sys.stderr)
        return 1

    for nome, serie in calcula_derivadas(bruto).items():
        bruto[nome] = serie

    # Eixo temporal comum em dias uteis: o BTC negocia 7 dias/semana e os
    # mercados tradicionais so em pregao. O ffill cobre feriados; nulos que
    # sobram no inicio sao mantidos (a serie ainda nao existia) e o dashboard
    # trata isso ao normalizar.
    grade = pd.bdate_range(inicio, bruto.index.max())
    bruto = bruto.reindex(bruto.index.union(grade)).ffill().reindex(grade)
    bruto = bruto.loc[bruto.index >= inicio]

    series_out, faltando = [], []
    for s in SERIES:
        col = s["id"]
        if col not in bruto.columns or bruto[col].notna().sum() == 0:
            faltando.append(s["label"])
            continue
        v = bruto[col]
        if s.get("escala"):
            v = v * s["escala"]
        series_out.append({
            "key": s["key"], "label": s["label"], "grupo": s["grupo"],
            "eixo": s["eixo"], "cor": s["cor"],
            "prefixo": s.get("prefixo", ""), "sufixo": s.get("sufixo", ""),
            "casas": s.get("casas", 2), "padrao": bool(s.get("padrao")),
            "fonte": s["fonte"],
            "valores": [None if pd.isna(x) else round(float(x), 6) for x in v],
        })

    grupos_usados = {s["grupo"] for s in series_out}
    payload = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inicio": bruto.index.min().strftime("%Y-%m-%d"),
        "fim": bruto.index.max().strftime("%Y-%m-%d"),
        "n": len(bruto),
        "datas": [d.strftime("%Y-%m-%d") for d in bruto.index],
        "grupos": [g for g in GRUPOS if g["id"] in grupos_usados],
        "series": series_out,
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(payload, ensure_ascii=False,
                                separators=(",", ":")), encoding="utf-8")

    print(f"\nOK  {payload['inicio']} -> {payload['fim']}  {payload['n']} dias úteis")
    print(f"    {len(series_out)} séries em {len(payload['grupos'])} grupos  "
          f"({SAIDA.stat().st_size / 1024:.0f} KB)")
    if faltando:
        print(f"    sem dados: {', '.join(faltando)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
