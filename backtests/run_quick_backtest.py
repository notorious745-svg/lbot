# backtests/run_quick_backtest.py
# Minimal working runner for XAUUSD 15m using Turtle/EMA votes + ATR stop.
# CLI: --minutes --symbol --session --strats --atr_n --atr_mult --max_layers --pyr_step_atr --vote --cooldown
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "XAUUSD_15m_clean.csv"
OUT  = ROOT / "backtests" / "out.txt"

def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df, n):
    return true_range(df).rolling(n, min_periods=n).mean()

def turtle_signals(close, w):
    hh = close.rolling(w).max().shift(1)
    ll = close.rolling(w).min().shift(1)
    long  = (close > hh).astype(int)
    short = (close < ll).astype(int)
    # ใช้เฉพาะ "เกิดสัญญาณใหม่" (cross) เพื่อลดสัญญาณซ้ำ
    long  = (long.diff() == 1).astype(int)
    short = (short.diff() == 1).astype(int)
    return long, short

def ema_signals(close):
    e10, e20 = ema(close,10), ema(close,20)
    long  = ((e10 > e20) & (e10.shift(1) <= e20.shift(1))).astype(int)
    short = ((e10 < e20) & (e10.shift(1) >= e20.shift(1))).astype(int)
    return long, short

def in_session(ts, session):
    # CSV มีคอลัมน์ 'time' เป็นสตริง; แปลงเป็น datetime แบบ naive
    hour = ts.hour
    if session == "ln_ny":
        # โซนกว้างๆ: 07:00–21:00 (UTC-ish) พอให้เทรดออกออเดอร์ได้
        return 7 <= hour <= 21
    return True  # "all" หรืออย่างอื่น → ไม่กรอง

def run(args):
    if not DATA.exists():
        print(f"[!] price file not found: {DATA}")
        sys.exit(2)

    df = pd.read_csv(DATA)
    # เลือกเฉพาะแท่งท้ายสุดตาม minutes (15m TF)
    bars = max(1000, int(args.minutes // 15))
    df = df.iloc[-bars:].copy()
    # เตรียมเวลา
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"].apply(lambda t: in_session(t, args.session))].copy()

    # ATR
    df["atr"] = atr(df, int(args.atr_n))

    # สร้างสัญญาณตาม strats
    strats = [s.strip().lower() for s in args.strats.split(",") if s.strip()]
    sig_long = pd.Series(0, index=df.index)
    sig_short= pd.Series(0, index=df.index)

    if "turtle20" in strats:
        l,s = turtle_signals(df["close"], 20); sig_long += l; sig_short += s
    if "turtle55" in strats:
        l,s = turtle_signals(df["close"], 55); sig_long += l; sig_short += s
    if "ema" in strats:
        l,s = ema_signals(df["close"]);       sig_long += l; sig_short += s

    vote = int(args.vote)
    want_long  = sig_long >= vote
    want_short = sig_short >= vote

    # backtest แบบ single position + ATR stop + cooldown
    trades = []
    pos = 0            # 0 none, +1 long, -1 short
    entry = 0.0
    entry_time = None
    cooldown_left = 0
    atr_mult = float(args.atr_mult)
    cooldown = int(args.cooldown)

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["close"]
        a = row["atr"]

        # update stop
        if pos != 0 and a > 0:
            if pos == 1:
                stop = entry - atr_mult * a
                if row["low"] <= stop:  # hit stop
                    exit_price = stop
                    trades.append((entry_time, row["time"], "long", entry, exit_price, exit_price - entry))
                    pos = 0; entry = 0.0; entry_time=None; cooldown_left = cooldown
                    continue
            else:
                stop = entry + atr_mult * a
                if row["high"] >= stop:
                    exit_price = stop
                    trades.append((entry_time, row["time"], "short", entry, exit_price, entry - exit_price))
                    pos = 0; entry = 0.0; entry_time=None; cooldown_left = cooldown
                    continue

        # cooldown
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        # exit on opposite vote
        if pos == 1 and want_short.iloc[i]:
            trades.append((entry_time, row["time"], "long", entry, price, price - entry))
            pos = 0; entry=0.0; entry_time=None; cooldown_left = cooldown
            continue
        if pos == -1 and want_long.iloc[i]:
            trades.append((entry_time, row["time"], "short", entry, price, entry - price))
            pos = 0; entry=0.0; entry_time=None; cooldown_left = cooldown
            continue

        # entry
        if pos == 0:
            if want_long.iloc[i]:
                pos = 1; entry = price; entry_time = row["time"]
            elif want_short.iloc[i]:
                pos = -1; entry = price; entry_time = row["time"]

    # ปิดปลายทางถ้ายังมีโพสิชัน
    if pos != 0 and entry_time is not None:
        last_time = df["time"].iloc[-1]
        last_px   = df["close"].iloc[-1]
        if pos == 1:
            trades.append((entry_time, last_time, "long", entry, last_px, last_px - entry))
        else:
            trades.append((entry_time, last_time, "short", entry, last_px, entry - last_px))

    # เขียนผล
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(trades, columns=["time_in","time_out","side","entry","exit","pnl"])
    tdf.to_csv(OUT, index=False)

    # คำนวณ metric แบบเร็ว (เผื่อ print_metrics ไม่รองรับ)
    pnl = tdf["pnl"].values if len(tdf) else np.array([])
    trades_n = int(len(tdf))
    sharpe = float(0 if trades_n==0 else (np.mean(pnl) / (np.std(pnl)+1e-9)) * np.sqrt(252))
    # max drawdown จาก equity
    eq = np.cumsum(pnl) if trades_n else np.array([0.0])
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    maxdd = float(0 if trades_n==0 else (np.max(dd) if len(dd) else 0.0))
    print(f"sharpe={sharpe:.6f}")
    print(f"maxdd={maxdd:.6f}")
    print(f"trades={trades_n}")

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, required=True)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--session", default="ln_ny", help="ln_ny|all")
    p.add_argument("--strats", default="turtle20")
    p.add_argument("--atr_n", type=int, default=20)
    p.add_argument("--atr_mult", type=float, default=3.0)
    p.add_argument("--max_layers", type=int, default=1)  # kept for CLI compat
    p.add_argument("--pyr_step_atr", type=float, default=1.0)  # ignored in this minimal runner
    p.add_argument("--vote", type=int, default=1)
    p.add_argument("--cooldown", type=int, default=0)
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args)
