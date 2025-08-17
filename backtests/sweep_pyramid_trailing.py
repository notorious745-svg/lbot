from __future__ import annotations
# --- make 'core' importable when running as a script ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------

import argparse, itertools, numpy as np, pandas as pd
from datetime import timedelta
from core.data_loader import load_price_csv
from core.entries import combined_signal
from core.position_manager import generate_position_series
from config import TAKER_FEE_BPS_PER_SIDE, ANN_FACTOR

def parse_list_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()!=""]

def parse_list_ints(s: str) -> list[int]:
    return [int(float(x)) for x in s.split(",") if x.strip()!=""]

def ann_sharpe(per_bar: pd.Series) -> float:
    r = per_bar.dropna()
    if len(r) < 2 or r.std() == 0: return 0.0
    return (r.mean()/r.std()) * np.sqrt(ANN_FACTOR)

def max_dd(per_bar: pd.Series) -> float:
    eq = (1+per_bar.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq/peak)-1
    return float(-dd.min()) if len(dd) else 0.0

def run_one(df, atr_n, atr_mult, step_atr, layers):
    sig = combined_signal(df)
    pos = generate_position_series(
        df, sig,
        atr_n=atr_n, atr_mult=atr_mult,
        pyramid_step_atr=step_atr, max_layers=layers
    )
    hold = pos.shift(1).fillna(0)
    ret  = df["close"].pct_change().fillna(0.0)
    gross = hold * ret
    fee = (TAKER_FEE_BPS_PER_SIDE/10000.0) * 2.0
    turns = hold.diff().abs().fillna(0)
    net = gross - (turns>0).astype(int)*fee

    trades = int((hold.diff().abs()>0).sum()//2)
    shp = ann_sharpe(net)
    mdd = max_dd(net)

    days = max((df["time"].iloc[-1] - df["time"].iloc[0]).days, 1)
    tpd = trades/days
    return shp, mdd, trades, tpd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=20000)
    ap.add_argument("--atr_n", type=int, default=14)
    ap.add_argument("--atr_mults", type=str, default="1.8,2.0,2.25,2.5,2.75,3.0,3.25")
    ap.add_argument("--steps_atr", type=str, default="0.6,0.8,1.0,1.2,1.4")
    ap.add_argument("--layers", type=str, default="0,1,2,3")
    ap.add_argument("--min_trades_per_day", type=float, default=1.0)
    ap.add_argument("--max_dd", type=float, default=0.30)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    df = load_price_csv()
    if args.minutes:
        end = df["time"].iloc[-1]
        start = end - timedelta(minutes=args.minutes)
        df = df[df["time"] >= start].reset_index(drop=True)

    atr_mults = parse_list_floats(args.atr_mults)
    steps     = parse_list_floats(args.steps_atr)
    layers    = parse_list_ints(args.layers)

    rows = []
    for am, st, ly in itertools.product(atr_mults, steps, layers):
        shp, mdd, trades, tpd = run_one(df, args.atr_n, am, st, ly)
        rows.append({
            "atr_mult": am, "step_atr": st, "layers": ly,
            "sharpe": round(shp,6), "maxdd": round(mdd,6),
            "trades": trades, "trades_per_day": round(tpd,2),
            "bars": len(df)
        })

    out = pd.DataFrame(rows)
    # filter & sort
    filt = (out["maxdd"] <= args.max_dd) & (out["trades_per_day"] >= args.min_trades_per_day)
    outf = out.loc[filt].sort_values(["sharpe","trades_per_day"], ascending=[False,False])
    if outf.empty:
        outf = out.sort_values("sharpe", ascending=False)

    # save & print
    path = ROOT / "backtests" / "sweep_pyramid_results.csv"
    outf.to_csv(path, index=False)
    print(outf.head(args.top).to_string(index=False))
    print(f"\nSaved: {path}")
