from __future__ import annotations
import pandas as pd
from datetime import time
from core.indicators import ema, bbands
from core.spike_filter import spike_flag

def sig_ema(df: pd.DataFrame) -> pd.Series:
    e10, e20, e50 = ema(df["close"], 10), ema(df["close"], 20), ema(df["close"], 50)
    s = pd.Series(0, index=df.index)
    s[(e10 > e20) & (e20 > e50)] = 1
    s[(e10 < e20) & (e20 < e50)] = -1
    return s

def _turtle_channel(series: pd.Series, n: int):
    hh = series.rolling(n, min_periods=n).max()
    ll = series.rolling(n, min_periods=n).min()
    return hh, ll

def sig_turtle(df: pd.DataFrame, n: int) -> pd.Series:
    hh, ll = _turtle_channel(df["close"], n)
    s = pd.Series(0, index=df.index)
    s[df["close"] > hh.shift(1)] = 1
    s[df["close"] < ll.shift(1)] = -1
    return s

def sig_meanrev(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    up, low, _ = bbands(df["close"], n=n, k=k)
    s = pd.Series(0, index=df.index)
    s[df["close"] > up]  = -1
    s[df["close"] < low] = 1
    return s

def session_mask_bkk(df: pd.DataFrame, mode: str | None = "ln_ny") -> pd.Series:
    """True = ในช่วงที่อนุญาต (เวลา BKK)"""
    if not mode or mode.lower() in ("none","all"):
        return pd.Series(True, index=df.index)
    t = df["time"].dt.time
    ln = (t >= time(13,0)) & (t <= time(22,30))         # London 13:00–22:30 BKK
    ny = (t >= time(20,30)) | (t <= time(4,0))          # New York 20:30–04:00 BKK (ข้ามวัน)
    return (ln | ny)

def combined_signal(
    df: pd.DataFrame,
    use_spike_mask: bool = True,
    session: str | None = "ln_ny",
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ema"]      = sig_ema(df)
    out["turtle20"] = sig_turtle(df, 20)
    out["turtle55"] = sig_turtle(df, 55)
    out["meanrev"]  = sig_meanrev(df, 20, 2.0)
    out["spike"]    = spike_flag(df)  # 1 = spike

    mask = pd.Series(True, index=df.index)
    if use_spike_mask:
        mask &= (out["spike"] == 0)
    mask &= session_mask_bkk(df, session)

    for c in ("ema","turtle20","turtle55","meanrev"):
        out[c] = out[c].where(mask, 0)

    return out
