# backtests/run_quick_backtest.py
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from datetime import timedelta
from core.data_loader import load_price_csv
from core.entries import combined_signal
from core.position_manager import generate_position_series
from config import TAKER_FEE_BPS_PER_SIDE, ANN_FACTOR

def sharpe_per_bar(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0: return 0.0
    return (r.mean()/r.std()) * np.sqrt(ANN_FACTOR)

def max_drawdown(series: pd.Series) -> float:
    eq = (1 + series.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq/peak) - 1.0
    return float(-dd.min()) if len(dd) else 0.0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, default=None)
    p.add_argument("--symbol", type=str, default="XAUUSD")
    # พารามิเตอร์พื้นฐาน (จูนได้ภายหลัง)
    p.add_argument("--atr_n", type=int, default=14)
    p.add_argument("--atr_mult", type=float, default=2.5)
    p.add_argument("--pyr_step_atr", type=float, default=1.0)
    p.add_argument("--max_layers", type=int, default=2)
    args, _ = p.parse_known_args()

    df = load_price_csv()
    if args.minutes:
        end = df["time"].iloc[-1]
        start = end - timedelta(minutes=args.minutes)
        df = df[df["time"] >= start].reset_index(drop=True)

    sig = combined_signal(df)
    pos = generate_position_series(
        df, sig,
        atr_n=args.atr_n,
        atr_mult=args.atr_mult,
        pyramid_step_atr=args.pyr_step_atr,
        max_layers=args.max_layers
    )

    out = pd.DataFrame(index=df.index)
    out["pos"] = pos.shift(1).fillna(0)  # ถือครองในบาร์ถัดไป
    out["ret"] = df["close"].pct_change().fillna(0.0)
    out["pnl_gross"] = out["pos"] * out["ret"]

    turns = out["pos"].diff().abs().fillna(0)
    fee = (TAKER_FEE_BPS_PER_SIDE/10000.0) * 2.0
    out["pnl_net"] = out["pnl_gross"] - (turns > 0).astype(int) * fee

    trades = int((out["pos"].diff().abs() > 0).sum() // 2)
    shp = sharpe_per_bar(out["pnl_net"])
    mdd = max_drawdown(out["pnl_net"])

    print(f"SYMBOL={args.symbol}")
    print(f"BARS={len(df)}")
    print(f"TRADES={trades}")
    print(f"SHARPE={shp:.6f}")
    print(f"MAXDD={mdd:.6f}")
