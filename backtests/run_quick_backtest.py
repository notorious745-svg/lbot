# backtests/run_quick_backtest.py
# Runner แบบ "ต้องมีเทรด": เขียน out.txt เป็นซีรีส์ต่อบาร์ไม่มี header, 9 คอลัมน์:
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

def build_votes(df: pd.DataFrame, strats: list[str], vote: int):
    L, S = [], []
    if "ema" in strats:
        e10, e20 = ema(df["close"], 10), ema(df["close"], 20)
        L.append(e10 > e20)
        S.append(e10 < e20)
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
        return z, z
    want_long  = pd.concat(L, axis=1).sum(axis=1)  >= vote
    want_short = pd.concat(S, axis=1).sum(axis=1) >= vote
    return want_long.fillna(False), want_short.fillna(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--session", default="all")          # ไม่กรองเวลา
    ap.add_argument("--strats", default="ema,turtle20,turtle55")
    ap.add_argument("--atr_n", type=int, default=20)     # ไม่ใช้ในเวอร์ชันนี้
    ap.add_argument("--atr_mult", type=float, default=0) # ปิดสต็อป
    ap.add_argument("--max_layers", type=int, default=1) # ใช้ไม้เดียว
    ap.add_argument("--pyr_step_atr", type=float, default=1.0)
    ap.add_argument("--vote", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=0)
    args = ap.parse_args()

    if not DATA.exists():
        raise FileNotFoundError(f"missing price file: {DATA}")

    df = pd.read_csv(DATA)
    bars = max(800, args.minutes // 15)
    df = df.iloc[-bars:].copy().reset_index(drop=True)

    want_long, want_short = build_votes(df, [s.strip().lower() for s in args.strats.split(",") if s.strip()], args.vote)
    # ดีบั๊ก: มีสัญญาณจริงไหม
    print(f"[dbg] bars={len(df)}  any_signals={(want_long|want_short).sum()}  L={int(want_long.sum())}  S={int(want_short.sum())}")

    pos = 0            # 0 flat, +1 long, -1 short
    equity = 1.0
    enter_long = enter_short = exit_pos = 0
    rows = []

    for i in range(len(df)):
        c = float(df["close"].iloc[i])
        # อัปเดต equity แบบต่อบาร์
        if i > 0:
            pprev = float(df["close"].iloc[i-1])
            if pos == 1:
                equity *= (c / pprev)
            elif pos == -1:
                equity *= (pprev / c)

        enter_long = enter_short = exit_pos = 0
        # logic ง่าย: เปลี่ยนฝั่งทันทีเมื่อคะแนนอีกฝั่งชนะ (มีเทรดแน่ๆ)
        if pos == 0:
            if want_long.iloc[i] and not want_short.iloc[i]:
                pos = 1; enter_long = 1
            elif want_short.iloc[i] and not want_long.iloc[i]:
                pos = -1; enter_short = 1
        else:
            if pos == 1 and want_short.iloc[i] and not want_long.iloc[i]:
                pos = 0; exit_pos = 1
                # เข้าฝั่งใหม่ทันที
                pos = -1; enter_short = 1
            elif pos == -1 and want_long.iloc[i] and not want_short.iloc[i]:
                pos = 0; exit_pos = 1
                pos = 1; enter_long = 1

        rows.append([i, c,
                     int(want_long.iloc[i]), int(want_short.iloc[i]),
                     enter_long, enter_short, exit_pos, pos, float(equity)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # เขียนแบบไม่มี header และคั่นด้วย comma ตามฟอร์แมตที่คุณมีอยู่
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"saved -> {OUT} rows={len(rows)}")
if __name__ == "__main__":
    main()
