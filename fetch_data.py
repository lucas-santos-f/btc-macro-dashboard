"""Download every series in sources.py and write web/data.json.

    python fetch_data.py             # last 7 years
    python fetch_data.py --years 10

The dashboard only reads the data.json written here, the browser makes no
network calls at all. This script is the only thing that talks to the outside.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sources import ADAPTERS, GROUPS, SERIES, derived_series

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "data.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, default=7)
    args = ap.parse_args()

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=args.years)
    print(f"window: {start.date()} -> {end.date()}\n")

    by_source = defaultdict(list)
    for s in SERIES:
        if s["source"] != "derived":
            by_source[s["source"]].append(s["id"])

    raw = pd.DataFrame()
    for source, ids in by_source.items():
        ids = list(dict.fromkeys(ids))
        print(f"  {source} ({len(ids)})...", flush=True)
        try:
            df = ADAPTERS[source](ids, start, end)
        except Exception as e:
            # one dead source shouldn't take the other 30 series down with it
            print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            print("    empty", file=sys.stderr)
            continue
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")]
        raw = df if raw.empty else raw.join(df, how="outer")

    if raw.empty:
        print("No source answered.", file=sys.stderr)
        return 1

    for name, s in derived_series(raw).items():
        raw[name] = s

    # BTC trades 7 days a week, everything else only on business days. Put it
    # all on a business-day grid; ffill covers holidays, leading NaNs stay
    # (the series didn't exist yet) and the dashboard handles them when rebasing
    grid = pd.bdate_range(start, raw.index.max())
    raw = raw.reindex(raw.index.union(grid)).ffill().reindex(grid)
    raw = raw.loc[raw.index >= start]

    series_out, missing = [], []
    for s in SERIES:
        col = s["id"]
        if col not in raw.columns or raw[col].notna().sum() == 0:
            missing.append(s["label"])
            continue
        v = raw[col]
        if s.get("scale"):
            v = v * s["scale"]
        series_out.append({
            "key": s["key"], "label": s["label"], "group": s["group"],
            "axis": s["axis"], "color": s["color"],
            "prefix": s.get("prefix", ""), "suffix": s.get("suffix", ""),
            "decimals": s.get("decimals", 2), "default": bool(s.get("default")),
            "source": s["source"],
            "values": [None if pd.isna(x) else round(float(x), 6) for x in v],
        })

    used_groups = {s["group"] for s in series_out}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": raw.index.min().strftime("%Y-%m-%d"),
        "end": raw.index.max().strftime("%Y-%m-%d"),
        "n": len(raw),
        "dates": [d.strftime("%Y-%m-%d") for d in raw.index],
        "groups": [g for g in GROUPS if g["id"] in used_groups],
        "series": series_out,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"\nok  {payload['start']} -> {payload['end']}  {payload['n']} business days")
    print(f"    {len(series_out)} series in {len(payload['groups'])} groups ({OUT.stat().st_size / 1024:.0f} KB)")
    if missing:
        print(f"    no data: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
