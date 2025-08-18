# backtests/run_quick_backtest.py
# Minimal runner ที่ "ต้องมีเทรด" แน่ๆ: ใช้ EMA10/20 + Turtle20/55 แบบ state (ไม่เอา diff)
# ปิด session filter ออกหมด, ปิด ATR stop ได้ด้วย --atr_mult 0

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

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def build_masks(df: pd.DataFrame, strats: list[str], vote: int):
    L, S = [], []
    if "ema" in strats:
        e10, e20 = ema(df["close"], 10), ema(df["close"], 20)
        L.append(e10 > e20)
        S.append(e10 < e20)
    if "turtle20" in strats:
        hh20 = df["close"].rolling(20).max().shift(1)
        ll20 = df["close"].rolling(20).min().shift(1)
        L.append(df["close"] > hh20)
        S.append(df["close"] < ll20)
    if "turtle55" in strats:
        hh55 = df["close"].rolling(55).max().shift(1)
        ll55 = df["close"].rolling(55).min().shift(1)
        L.append(df["close"] > hh55)
        S.append(df["close"] < ll55)

    if not L:
        z = pd.Series(False, index=df.index)
        return z, z

    want_long  = pd.concat(L, axis=1).sum(axis=1) >= vote
    want_short = pd.concat(S, axis=1).sum(axis=1) >= vote
    return want_long.fillna(False), want_short.fillna(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--session", default="all")          # ไม่กรองเวลา
    ap.add_argument("--strats", default="ema,turtle20,turtle55")
    ap.add_argument("--atr_n", type=int, default=20)
    ap.add_argument("--atr_mult", type=float, default=0) # ค่าเริ่ม = 0 (ปิด stop เพื่อให้มีเทรดแน่ๆ)
    ap.add_argument("--max_layers", type=int, default=1) # ไม่ใช้พีระมิด
    ap.add_argument("--pyr_step_atr", type=float, default=1.0)
    ap.add_argument("--vote", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=0)
    args = ap.parse_args()

    if not DATA.exists():
        raise FileNotFoundError(f"missing price file: {DATA}")

    df = pd.read_csv(DATA)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    bars = max(500, args.minutes // 15)   # อย่างน้อย 500 แท่ง
    df = df.iloc[-bars:].copy()
    df["atr"] = atr(df, args.atr_n).bfill()

    strats = [s.strip().lower() for s in args.strats.split(",") if s.strip()]
    want_long, want_short = build_masks(df, strats, args.vote)

    # debug สั้น ๆ ให้เห็นว่ามีสัญญาณจริง
    print(f"[dbg] bars={len(df)}  long_votes={int(want_long.sum())}  short_votes={int(want_short.sum())}  any={(want_long|want_short).sum()}")

    pos = 0
    entry = 0.0
    entry_t = None
    cd = 0
    trades = []

    for i in range(len(df)):
        px = float(df["close"].iloc[i])
        t  = df["time"].iloc[i] if "time" in df.columns else i
        a  = float(df["atr"].iloc[i]) if not np.isnan(df["atr"].iloc[i]) else 0.0

        if cd > 0:
            cd -= 1
            continue

        if pos == 0:
            if want_long.iloc[i] and not want_short.iloc[i]:
                pos, entry, entry_t = 1, px, t
            elif want_short.iloc[i] and not want_long.iloc[i]:
                pos, entry, entry_t = -1, px, t
        else:
            exit_now = False
            exit_px = px

            # ออกเมื่อได้สัญญาณฝั่งตรงข้าม
            if pos == 1 and (want_short.iloc[i] and not want_long.iloc[i]):
                exit_now = True
            elif pos == -1 and (want_long.iloc[i] and not want_short.iloc[i]):
                exit_now = True

            # ATR stop (เปิดได้ด้วย --atr_mult > 0)
            if not exit_now and args.atr_mult > 0 and a > 0:
                if pos == 1:
                    stop = entry - args.atr_mult * a
                    if df["low"].iloc[i] <= stop:
                        exit_now, exit_px = True, stop
                else:
                    stop = entry + args.atr_mult * a
                    if df["high"].iloc[i] >= stop:
                        exit_now, exit_px = True, stop

            if exit_now:
                pnl = exit_px - entry if pos == 1 else entry - exit_px
                trades.append((entry_t, t, "long" if pos==1 else "short", entry, exit_px, pnl))
                pos, entry, entry_t, cd = 0, 0.0, None, args.cooldown

    if pos != 0 and entry_t is not None:  # ปิดไม้ค้างที่แท่งสุดท้าย
        last_t = df["time"].iloc[-1] if "time" in df.columns else len(df)-1
        last_px = float(df["close"].iloc[-1])
        pnl = last_px - entry if pos == 1 else entry - last_px
        trades.append((entry_t, last_t, "long" if pos==1 else "short", entry, last_px, pnl))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(trades, columns=["time_in","time_out","side","entry","exit","pnl"])
    tdf.to_csv(OUT, index=False)
    print(f"trades={len(tdf)}")  # ให้เห็นตรง ๆ
