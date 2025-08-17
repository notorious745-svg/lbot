$spike = @'
from __future__ import annotations
import pandas as pd
from core.indicators import atr

def spike_flag(df: pd.DataFrame, n: int = 14, k: float = 3.0) -> pd.Series:
    """ธง spike = 1 เมื่อช่วง high-low ใหญ่กว่า k*ATR; ใช้ปิดสัญญาณชั่วคราว"""
    rng = (df["high"] - df["low"]).abs()
    a = atr(df, n=n)
    return ((rng > (k * a)).astype(int)).reindex(df.index).fillna(0)
'@
Set-Content -Encoding utf8 core\spike_filter.py $spike
