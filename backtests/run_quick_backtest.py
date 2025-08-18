# backtests/run_quick_backtest.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "XAUUSD_15m_clean.csv"
OUT  = ROOT / "backtests" / "out.txt"

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, min_periods=n).mean()

def turtle(df: pd.DataFrame, n: int):
    hh = df["close"].rolling(n).max().shift(1)
    ll = df["close"].rolling(n).min().shift(1)
    return (df["close"] > hh), (df["close"] < ll)

def compute_indicators(df: pd.DataFrame):
    for n in (10, 20, 50):
        df[f"ema{n}"] = ema(df["close"], n)
    df["ema_long"]  = (df["ema10"] > df["ema20"]) & (df["ema20"] > df["ema50"])
    df["ema_short"] = (df["ema10"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    t20l, t20s = turtle(df, 20)
    t55l, t55s = turtle(df, 55)
    df["t20_long"], df["t20_short"] = t20l, t20s
    df["t55_long"], df["t55_short"] = t55l, t55s
    return df

def pick_strats_mask(df: pd.DataFrame, strats: list[str]) -> tuple[pd.Series, pd.Series]:
    longs = []
    shorts = []
    for s in strats:
        s = s.strip().lower()
        if s == "ema":
            longs.append(df["ema_long"]);  shorts.append(df["ema_short"])
        elif s == "turtle20":
            longs.append(df["t20_long"]);  shorts.append(df["t20_short"])
        elif s == "turtle55":
            longs.append(df["t55_long"]);  shorts.append(df["t55_short"])
    if not longs:
        # safety: ถ้าไม่รู้จัก strat ใด ๆ ให้ถือว่าไม่มีสัญญาณ
        z = pd.Series(False, index=df.index)
        return z, z
    return pd.concat(longs, axis=1).sum(axis=1), pd.concat(shorts, axis=1).sum(axis=1)

def backtest(df: pd.DataFrame, strats: list[str], vote: int, cooldown: int) -> pd.DataFrame:
    """
    กติกาง่าย ๆ:
      - เปิด Long ถ้า vote_long >= vote และมากกว่า vote_short
      - เปิด Short ถ้า vote_short >= vote และมากกว่า vote_long
      - ปิดเมื่อเจอสัญญาณฝั่งตรงข้าม
      - ใช้ราคา close แท่งสัญญาณ (ไม่ intrabar)
    """
    vote_long, vote_short = pick_strats_mask(df, strats)
    pos = 0   # 0=flat, 1=long, -1=short
    entry_px = 0.0
    entry_t  = None
    trades = []

    # กันกรณี cooldown>0 แล้วเปิดติด ๆ กัน (ที่นี่ใช้เป็น post-entry freeze)
    cool = 0

    for i in range(len(df)):
        c = float(df["close"].iloc[i])
        ts = df["time"].iloc[i] if "time" in df.columns else i

        if pos == 0:
            if cool > 0:
                cool -= 1
            else:
                if vote_long.iloc[i] >= vote and vote_long.iloc[i] > vote_short.iloc[i]:
                    pos, entry_px, entry_t = 1, c, ts
                    cool = cooldown
                elif vote_short.iloc[i] >= vote and vote_short.iloc[i] > vote_long.iloc[i]:
                    pos, entry_px, entry_t = -1, c, ts
                    cool = cooldown
        else:
            if pos == 1 and vote_short.iloc[i] >= vote and vote_short.iloc[i] > vote_long.iloc[i]:
                pnl = (c - entry_px) / entry_px
                trades.append((entry_t, ts, "LONG", entry_px, c, pnl))
                pos = 0
            elif pos == -1 and vote_long.iloc[i] >= vote and vote_long.iloc[i] > vote_short.iloc[i]:
                pnl = (entry_px - c) / entry_px
                trades.append((entry_t, ts, "SHORT", entry_px, c, pnl))
                pos = 0

    # force close ที่แท่งสุดท้าย ถ้ายังถืออยู่
    if pos != 0:
        c = float(df["close"].iloc[-1])
        ts = df["time"].iloc[-1] if "time" in df.columns else len(df)-1
        if pos == 1:
            pnl = (c - entry_px) / entry_px
            trades.append((entry_t, ts, "LONG", entry_px, c, pnl))
        else:
            pnl = (entry_px - c) / entry_px
            trades.append((entry_t, ts, "SHORT", entry_px, c, pnl))

    # สร้าง equity จากผลตอบแทนต่อดีล (คูณต่อเนื่อง)
    if trades:
        tdf = pd.DataFrame(trades, columns=["entry_time","exit_time","side","entry","exit","ret"])
        tdf["equity"] = (1.0 + tdf["ret"]).cumprod()
    else:
        tdf = pd.DataFrame(columns=["entry_time","exit_time","side","entry","exit","ret","equity"])
    return tdf

def slice_minutes(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes is None or minutes <= 0:
        return df
    bars = max(1, minutes // 15)  # M15 → 15 นาทีต่อแท่ง
    return df.iloc[-bars:].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--symbol", type=str, default="X
