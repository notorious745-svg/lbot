from __future__ import annotations
import pandas as pd

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def bbands(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = ma + k*sd, ma - k*sd
    return upper, lower, ma
