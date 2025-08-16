# core/entries.py
from __future__ import annotations
from typing import Dict, Literal

Signal = Literal[0, 1, 2]  # 0=hold, 1=open/add long, 2=exit-all (ใช้ในอนาคต)

def ema_trend_long(row: Dict[str, float]) -> bool:
    """
    ใช้ EMA50 เหนือ EMA200 + close เหนือ EMA50 = แนวโน้มขาขึ้น
    """
    return (
        row.get("ema_fast", 0.0) > row.get("ema_slow", 0.0)
        and row.get("close", 0.0) > row.get("ema_fast", 0.0)
    )

def spike_filter(row: Dict[str, float], k: float) -> bool:
    """
    true = เป็น spike (ควรหลีกเลี่ยง)
    วัดจาก body/ATR > k
    """
    atr = max(row.get("atr", 1e-9), 1e-9)
    body = abs(row.get("close", 0.0) - row.get("open", 0.0))
    return (body / atr) > k

def decide_entry(
    row: Dict[str, float],
    *,
    bar_close_only: bool = True,
    use_spike: bool = True,
    spike_k: float = 2.2,
) -> Signal:
    """
    return: 0=hold, 1=open/add long
    """
    if use_spike and spike_filter(row, spike_k):
        return 0
    if bar_close_only and not row.get("bar_closed", True):
        return 0
    return 1 if ema_trend_long(row) else 0
