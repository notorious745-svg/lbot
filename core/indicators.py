from __future__ import annotations
import numpy as np
import pandas as pd

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=1).mean()

def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    ma = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    up = ma + k * sd
    low = ma - k * sd
    return up, low, ma

def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(n, min_periods=n).mean()
