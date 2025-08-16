from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import config

@dataclass
class PosState:
    side: int = 0      # 0=flat, +1=long
    entry: float = 0.0
    qty: int = 0
    sl: Optional[float] = None
    group_id: int = 0

class PositionManager:
    def __init__(self, contract_size: float = 1.0):
        self.state = PosState()
        self.contract = float(contract_size)
        self.open_groups = 0
        self.daily_risk_used = 0.0
        self.cur_day = None

    def new_day(self, equity: float, day) -> None:
        if self.cur_day != day:
            self.cur_day = day
            self.daily_risk_used = 0.0

    def risk_per_trade_value(self, equity: float) -> float:
        return equity * config.RISK_PER_TRADE

    def can_add_long(self, px: float) -> bool:
        if self.state.side == 0:
            return True
        if self.state.side == +1 and px > self.state.entry:
            return True
        return False

    def open_or_add_long(self, px: float, atr: float, equity: float) -> None:
        risk_val = self.risk_per_trade_value(equity)
        sl_buf = max(atr * 1.5, 0.5)
        stop_px = px - sl_buf
        per_unit_risk = max(px - stop_px, 1e-4) * self.contract
        qty = max(int(risk_val / per_unit_risk), 1)

        if self.state.side == 0:
            self.state = PosState(side=+1, entry=px, qty=qty, sl=stop_px, group_id=self.open_groups)
        else:
            self.state.qty += qty  # pyramiding เฉพาะตอนกำไร (เช็คก่อนเรียกแล้ว)
        self.daily_risk_used += config.RISK_PER_TRADE * equity

    def mtm(self, px: float) -> float:
        if self.state.side == 0:
            return 0.0
        return self.state.qty * (px - self.state.entry) * self.contract

    def flat(self, px: float) -> float:
        if self.state.side == 0:
            return 0.0
        pnl = self.mtm(px)
        self.state = PosState()
        return pnl
