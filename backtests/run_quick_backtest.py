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

def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([(df["high"]-df["low"]).abs(),
                      (df["high"]-pc).abs(),
                      (df["low"] -pc).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return true_range(df).rolling(n, min_periods=n).mean()

def votes_counts(df: pd.DataFrame, strats: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    L, S = [], []
    e10, e20 = ema(df["close"], 10), ema(df["close"], 20)
    ema_bias = (e10 > e20).astype(int) - (e10 < e20).astype(int)

    if "ema" in strats:
        L.append(e10 > e20); S.append(e10 < e20)
    if "turtle20" in strats:
        hh, ll = df["close"].rolling(20).max().shift(1), df["close"].rolling(20).min().shift(1)
        L.append(df["close"] > hh); S.append(df["close"] < ll)
    if "turtle55" in strats:
        hh, ll = df["close"].rolling(55).max().shift(1), df["close"].rolling(55).min().shift(1)
        L.append(df["close"] > hh); S.append(df["close"] < ll)

    if not L:
        z = pd.Series(0, index=df.index)
        return z, z, ema_bias

    long_cnt  = pd.concat(L, axis=1).sum(axis=1).fillna(0).astype(int)
    short_cnt = pd.concat(S, axis=1).sum(axis=1).fillna(0).astype(int)
    return long_cnt, short_cnt, ema_bias

def decide_winner(long_cnt, short_cnt, ema_bias, min_votes: int) -> pd.Series:
    # ฝั่งชนะ: +1 / -1 / 0 โดยต้องมีคะแนนฝั่งชนะ >= min_votes
    diff = long_cnt - short_cnt
    win = pd.Series(0, index=diff.index)
    win[diff > 0] = 1
    win[diff < 0] = -1
    tie = (diff == 0)
    win[tie] = ema_bias[tie]
    # ต้องถึงเกณฑ์โหวตขั้นต่ำ
    win[(win == 1) & (long_cnt < min_votes)] = 0
    win[(win == -1) & (short_cnt < min_votes)] = 0
    return win.astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--session", default="all")
    ap.add_argument("--strats", default="ema,turtle20,turtle55")
    ap.add_argument("--atr_n", type=int, default=20)
    ap.add_argument("--atr_mult", type=float, default=0.0)   # 0 = ปิดสต็อป
    ap.add_argument("--max_layers", type=int, default=1)     # ใช้ไม้เดียว
    ap.add_argument("--pyr_step_atr", type=float, default=1.0)
    ap.add_argument("--vote", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=0)       # แท่ง
    args = ap.parse_args()

    if not DATA.exists():
        raise FileNotFoundError(f"missing price file: {DATA}")

    df = pd.read_csv(DATA)
    bars = max(800, args.minutes // 15)
    df = df.iloc[-bars:].copy().reset_index(drop=True)
    df["atr"] = atr(df, args.atr_n).bfill()

    strats = [s.strip().lower() for s in args.strats.split(",") if s.strip()]
    long_cnt, short_cnt, ema_bias = votes_counts(df, strats)
    winner = decide_winner(long_cnt, short_cnt, ema_bias, min_votes=args.vote)

    print(f"[dbg] bars={len(df)} any_signal={(winner!=0).sum()} long_win={(winner==1).sum()} short_win={(winner==-1).sum()}")

    pos = 0            # 0 flat, +1 long, -1 short
    entry = 0.0
    equity = 1.0
    cooldown_left = 0
    enter_long = enter_short = exit_pos = 0
    rows = []
    enters = exits = 0
    a_mult = float(args.atr_mult)

    for i in range(len(df)):
        c  = float(df["close"].iloc[i])
        a  = float(df["atr"].iloc[i]) if not np.isnan(df["atr"].iloc[i]) else 0.0

        # update equity ต่อบาร์
        if i > 0:
            pprev = float(df["close"].iloc[i-1])
            if pos == 1:   equity *= (c / pprev)
            elif pos == -1: equity *= (pprev / c)

        enter_long = enter_short = exit_pos = 0

        # ถ้ามีโพสิชัน → เช็ค ATR stop ก่อน (สต็อปทำงานแม้อยู่ช่วง cooldown)
        if pos != 0 and a_mult > 0 and a > 0:
            if pos == 1:
                stop = entry - a_mult * a
                if df["low"].iloc[i] <= stop:
                    exit_pos = 1; exits += 1
                    pos = 0; entry = 0.0
                    cooldown_left = args.cooldown
            else:
                stop = entry + a_mult * a
                if df["high"].iloc[i] >= stop:
                    exit_pos = 1; exits += 1
                    pos = 0; entry = 0.0
                    cooldown_left = args.cooldown

        # ถ้าหมดคูลดาวน์แล้วค่อยพิจารณา entry/flip
        if cooldown_left > 0:
            cooldown_left -= 1
        else:
            want = int(winner.iloc[i])
            # flip หรือเข้าครั้งแรกเมื่อมีสัญญาณ
            if want == 1 and pos <= 0:
                if pos == -1: exit_pos = 1; exits += 1
                pos = 1; entry = c; enter_long = 1; enters += 1
                cooldown_left = args.cooldown
            elif want == -1 and pos >= 0:
                if pos == 1: exit_pos = 1; exits += 1
                pos = -1; entry = c; enter_short = 1; enters += 1
                cooldown_left = args.cooldown

        rows.append([i, c,
                     int(long_cnt.iloc[i] > 0), int(short_cnt.iloc[i] > 0),
                     enter_long, enter_short, exit_pos, pos, float(equity)])

    # ปิดไม้ค้างปลายทาง
    if pos != 0:
        rows[-1][6] = 1
        exits += 1
        pos = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="
