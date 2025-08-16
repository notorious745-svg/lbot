from __future__ import annotations
import pandas as pd
from indicators import atr

def spike_flag(df: pd.DataFrame, n_atr: int = 14, k: float = 3.0) -> pd.Series:
    a = atr(df, n_atr)
    rng = (df["high"] - df["low"]).abs()
    return (rng > k * a.shift(1)).astype(int)  # ใช้ ATR ของแท่งก่อนหน้าเป็นเกณฑ์
