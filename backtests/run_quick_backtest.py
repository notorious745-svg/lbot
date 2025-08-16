from __future__ import annotations
import numpy as np
import pandas as pd
from core.data_loader import load_price_csv
from core.entries import combined_signal
from core.position_manager import apply_caps
from config import TAKER_FEE_BPS_PER_SIDE, MIN_TRADES_PER_DAY, ANN_FACTOR

def simulate(df: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    out = df[["time","close"]].copy()
    pos = apply_caps(sig)
    out["pos"] = pos["sum"].shift(1).fillna(0)
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["pnl_gross"] = out["pos"] * out["ret"]

    turns = pos["sum"].diff().abs().fillna(0)
    fee = (TAKER_FEE_BPS_PER_SIDE / 10000.0) * 2.0
    out["pnl_net"] = out["pnl_gross"] - (turns > 0).astype(int) * fee
    return out

def sharpe_per_bar(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0: return 0.0
    return (r.mean() / r.std()) * np.sqrt(ANN_FACTOR)

if __name__ == "__main__":
    df  = load_price_csv()
    sig = combined_signal(df)
    res = simulate(df, sig)

    switches = res["pos"].diff().abs().fillna(0)
    trades = int((switches > 0).sum() / 2)
    days = max((df["time"].iloc[-1] - df["time"].iloc[0]).days, 1)
    tpd  = trades / days
    shp  = sharpe_per_bar(res["pnl_net"])

    print(f"[i] days={days} trades={trades} trades/day={tpd:.2f} sharpe≈{shp:.2f}")
    if tpd < MIN_TRADES_PER_DAY:
        print(f"[!] trades/day < {MIN_TRADES_PER_DAY}  (config.py)")
    print("[ok] quick backtest done (demo-safe if CSV missing)")
