from __future__ import annotations
import numpy as np

def step_reward(
    step_pnl: float,
    equity_curve: np.ndarray,
    max_dd_limit: float = 0.2,
    trade_cost: float = 0.0,
) -> float:
    """
    Reward ราย step:
      + ผลตอบแทนรายแท่ง (หลังหัก cost)
      + โทษเมื่อเข้าเขต drawdown หนัก (นิ่ม ๆ ไม่แกว่ง)
    """
    r = step_pnl - trade_cost
    if equity_curve is None or len(equity_curve) < 2:
        return r

    peak = np.max(equity_curve)
    dd = 0.0 if peak <= 0 else 1.0 - (equity_curve[-1] / peak)
    if dd > max_dd_limit:
        # โทษแบบนิ่ม (soft penalty)
        r -= 0.5 * (dd - max_dd_limit)
    return float(r)
