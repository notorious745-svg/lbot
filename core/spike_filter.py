from __future__ import annotations

import pandas as pd
from core.indicators import atr  # อ้างอิงฟังก์ชัน ATR ที่มีในโปรเจกต์


def spike_flag(df: pd.DataFrame, n: int = 14, k: float = 3.0) -> pd.Series:
    """
    คืนค่า Series 1/0 ระบุว่าแท่งนั้น 'กว้าง' เกิน k * ATR (ถือเป็น spike)
    - n: ช่วงคำนวณ ATR
    - k: เกณฑ์เท่าของ ATR
    """
    a = atr(df, n)  # ต้องมีคอลัมน์ high/low/close
    rng = (df["high"] - df["low"]).abs()
    flag = (rng > (k * a)).astype(int)
    return flag.reindex(df.index).fillna(0)
