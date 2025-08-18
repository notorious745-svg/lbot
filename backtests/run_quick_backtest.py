# backtests/run_quick_backtest.py
# out.txt: ต่อบาร์ 9 คอลัมน์ (ไม่มี header):
# idx, close, want_long, want_short, enter_long, enter_short, exit_pos, pos, equity
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "XAUUSD_15m_clean.csv"
OUT  = ROOT / "backtests" / "out.txt"

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def votes_counts(df: pd.DataFrame, strats: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """คืน (long_cnt, short_cnt, ema_bias) ต่อบาร์"""
    L, S = [], []
    ema10, ema20 = ema(df["close"], 10), ema(df["close"], 20)
    ema_bias = (ema10 > ema20).astype(int) - (ema10 < ema20).astype(int)

    if "ema" in strats:
        L.append(ema10 > ema20)
        S.append(ema10 < ema20)
    if "turtle20" in strats:
        hh = df["close"].rolling(20).max().shift(1)
        ll = df["close"].rolling(20).min().shift(1)
        L.append(df["close"] > hh)
        S.append(df["close"] < ll)
    if "turtle55" in strats:
        hh = df["close"].rolling(55).max().shift(1)
        ll = df["close"].rolling(55).min().shift(1)
        L.append(df["close"] > hh)
        S.append(df["close"] < ll)

    if not L:
        z = pd.Series(False, index=df.index)
        return z.astype(int), z.astype(int), ema_bias

    long_cnt  = pd.concat(L, axis=1).sum(axis=1).astype(int)
    short_cnt = pd.concat(S, axis=1).sum(axis=1).astype(int)
    return long_cnt.fillna(0).astype(int), short_cnt.fillna(0).astype(int), ema_bias

def decide_side(long_cnt, short_cnt, ema_bias):
    """winner: +1 long, -1 short, 0 no trade; tie ใช้ ema_bias เป็นตัวตัดสิน"""
    diff = (long_cnt - short_cnt)
    winner = diff.copy()
    winner[diff > 0] = 1
    winner[diff < 0] = -1
    # tie -> ใช้ ema_bias (ถ้า 0 ด้วย ให้คง 0)
    tie = (diff == 0)
    winner[tie] = ema_bias[tie]
    return winner.astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--session", default="all")          # ไม่กรองเวลา
    ap.add_argument("--strats", default="ema,turtle20,turtle55")
    ap.add_argument("--atr_n", type=int, default=20)
    ap.add_argument("--atr_mult", type=float, default=0) # ปิด stop
    ap.add_argument("--max_layers", type=int, default=1) # ไม้เดียว
    ap.add_argument("--pyr_step_atr", type=float, default=1.0)
    ap.add_argument("--vote", type=int, default=1)       # ใช้เป็น minimum count ที่ต้องมีอย่างน้อยในฝั่งที่ชนะ
    ap.add_argument("--cooldown", type=int, default=0)
    args = ap.parse_args()

    if not DATA.exists():
        raise FileNotFoundError(f"missing price file: {DATA}")

    df = pd.read_csv(DATA)
    bars = max(800, args.minutes // 15)
    df = df.iloc[-bars:].copy().reset_index(drop=True)

    strats = [s.strip().lower() for s in args.strats.split(",") if s.strip()]
    long_cnt, short_cnt, ema_bias = votes_counts(df, strats)
    winner = decide_side(long_cnt, short_cnt, ema_bias)

    # ต้องมีอย่างน้อย vote ในฝั่งที่ชนะ
    valid = ((long_cnt.where(winner==1, 0) >= args.vote) |
             (short_cnt.where(winner==-1, 0) >= args.vote))

    print(f"[dbg] bars={len(df)} any_signal={(winner!=0).sum()}  long_win={(winner==1).sum()}  short_win={(winner==-1).sum()}")

    pos = 0     # 0 flat, +1 long, -1 short
    equity = 1.0
    enter_long = enter_short = exit_pos = 0
    rows = []
    enters = exits = 0

    for i in range(len(df)):
        c = float(df["close"].iloc[i])

        # update equity ต่อบาร์
        if i > 0:
            pprev = float(df["close"].iloc[i-1])
            if pos == 1:
                equity *= (c / pprev)
            elif pos == -1:
                equity *= (pprev / c)

        enter_long = enter_short = exit_pos = 0

        if valid.iloc[i]:
            want = int(winner.iloc[i])  # +1 / -1
            if pos == 0:
                if want == 1:
                    pos = 1; enter_long = 1; enters += 1
                elif want == -1:
                    pos = -1; enter_short = 1; enters += 1
            else:
                if want == 1 and pos == -1:
                    exit_pos = 1; exits += 1
                    pos = 1; enter_long = 1; enters += 1
                elif want == -1 and pos == 1:
                    exit_pos = 1; exits += 1
                    pos = -1; enter_short = 1; enters += 1
        # ถ้าไม่ valid → คงสถานะ

        rows.append([i, c,
                     int(long_cnt.iloc[i] > 0), int(short_cnt.iloc[i] > 0),
                     enter_long, enter_short, exit_pos, pos, float(equity)])

    # ปิดไม้ค้างที่แท่งสุดท้าย (ให้นับ exit)
    if pos != 0:
        rows[-1][6] = 1
        exits += 1
        pos = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"[dbg2] enters={enters} exits={exits}  saved->{OUT} rows={len(rows)}")

if __name__ == "__main__":
    main()
