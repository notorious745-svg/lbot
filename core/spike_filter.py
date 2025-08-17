from __future__ import annotations

import pandas as pd


def spike_flag(df: pd.DataFrame, n: int = 14, k: float = 3.0) -> pd.Series:
    """
    ธง spike = 1 เมื่อช่วง high-low เกิน k * ATR(n)
    มิฉะนั้น 0
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    rng = (high - low).abs()
    # ATR แบบง่าย: rolling mean ของ true range (ที่นี่ใช้ high-low เป็นตัวแทน)
    atr = rng.rolling(n, min_periods=1).mean()

    flag = (rng > (k * atr)).astype(int)
    return flag.reindex(df.index).fillna(0)
