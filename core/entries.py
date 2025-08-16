from __future__ import annotations
import pandas as pd
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

def sig_meanrev(df: pd.DataFrame) -> pd.Series:
    up, low, ma = bbands(df["close"], n=20, k=2.0)
    s = pd.Series(0, index=df.index)
    s[df["close"] > up]  = -1
    s[df["close"] < low] = 1
    return s

def combined_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ema"]      = sig_ema(df)
    out["turtle20"] = sig_turtle(df, 20)
    out["turtle55"] = sig_turtle(df, 55)
    out["meanrev"]  = sig_meanrev(df)
    out["spike"]    = spike_flag(df)  # 1 = spike, หลีกเลี่ยงเข้าไม้
    return out
