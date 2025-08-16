# core/position_manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Position:
    entry: float
    sl: float
    size: float      # lots/units (เชิงนามธรรม: ใช้คูณกับ PnL แทน)
    added_times: int = 0

@dataclass
class PMConfig:
    risk_per_trade: float
    daily_risk_cap: float
    max_pyramid: int
    add_on_gain_r: float
    breakeven_after_r: float
    atr_mult: float
    pip: float

@dataclass
class PositionManager:
    cfg: PMConfig
    equity: float
    day_risk_used: float = 0.0
    positions: List[Position] = field(default_factory=list)

    # ==== sizing / stop ====
    def _risk_amount(self) -> float:
        # จำกัดต่อวันด้วย
        remain = max(self.cfg.daily_risk_cap - self.day_risk_used, 0.0)
        return max(min(self.equity * self.cfg.risk_per_trade, self.equity * remain), 0.0)

    def calc_sl_and_size(self, price: float, atr: float) -> tuple[float, float]:
        sl = price - self.cfg.atr_mult * atr
        risk_per_unit = max(price - sl, self.cfg.pip)  # 1R = distance
        risk_amt = self._risk_amount()
        size = 0.0 if risk_per_unit <= 0 else risk_amt / risk_per_unit
        return sl, size

    # ==== open / add ====
    def can_add(self) -> bool:
        return len(self.positions) == 0 or (
            len(self.positions) > 0 and self.positions[-1].added_times < self.cfg.max_pyramid
        )

    def try_open_or_add(self, price: float, atr: float):
        if not self.can_add():
            return
        sl, size = self.calc_sl_and_size(price, atr)
        if size <= 0:
            return
        if len(self.positions) == 0:
            self.positions.append(Position(entry=price, sl=sl, size=size, added_times=0))
            self.day_risk_used += self.cfg.risk_per_trade
        else:
            last = self.positions[-1]
            # add เมื่อกำไรถึงเกณฑ์
            r_gain = (price - last.entry) / max((last.entry - last.sl), self.cfg.pip)
            if r_gain >= self.cfg.add_on_gain_r and last.added_times < self.cfg.max_pyramid:
                self.positions.append(Position(entry=price, sl=sl, size=size, added_times=last.added_times + 1))
                self.day_risk_used += self.cfg.risk_per_trade

    # ==== update sl (breakeven + trailing) ====
    def update_sl(self, price: float, atr: float):
        for p in self.positions:
            r_gain = (price - p.entry) / max((p.entry - p.sl), self.cfg.pip)
            if r_gain >= self.cfg.breakeven_after_r:
                p.sl = max(p.sl, p.entry)          # BE
            trail_sl = price - self.cfg.atr_mult * atr
            p.sl = max(p.sl, trail_sl)             # ATR trail

    # ==== check exit ====
    def stop_out(self, low: float) -> bool:
        # ถ้าราคา low ตัด SL ใด ๆ -> ปิดทุกไม้
        return any(low <= p.sl for p in self.positions)

    def exit_all(self, price: float) -> float:
        """ปิดทุกไม้ คืน PnL"""
        pnl = 0.0
        for p in self.positions:
            pnl += (price - p.entry) * p.size
        self.positions.clear()
        return pnl

    # ==== mark-to-market ====
    def unrealized(self, price: float) -> float:
        return sum((price - p.entry) * p.size for p in self.positions)
