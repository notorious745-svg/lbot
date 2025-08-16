from __future__ import annotations
from typing import Literal, Dict

Signal = Literal[0, 1, 2]  # 0=hold, 1=open/add long, 2=exit-all

def ema_trend_long(row: Dict[str, float]) -> bool:
    # เทรนด์ขึ้น: close เหนือ EMA50 และ EMA10 > EMA20
    return (row["close"] > row["ema_trend"]) and (row["ema_fast"] > row["ema_slow"])

def decide_entry(row: Dict[str, float], bar_close_only: bool = True) -> Signal:
    """
    ใช้เฉพาะตอนแท่งปิด (bar_close_only=True)
    เคารพ spike filter: ถ้า is_spike=True → hold
    """
    if row.get("is_spike", False):
        return 0
    if not bar_close_only:
        # เผื่อในอนาคต—but default ของเรา = True
        pass
    return 1 if ema_trend_long(row) else 0
