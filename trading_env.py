from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import config
from features import build_features
from reward_fn import step_reward

@dataclass
class Position:
    side: int = 0      # 0=flat, +1=long, -1=short (เริ่มจาก long-only ได้)
    entry: float = 0.0
    qty: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    group_id: Optional[int] = None  # ใช้เวลา pyramiding / exit พร้อมกัน

class TradingEnv:
    """
    Minimal on-close-bar environment สำหรับ XAUUSD M15
    action space:
      0 = hold, 1 = open/add long, 2 = exit all (flat)
    """
    def __init__(self, df_raw: pd.DataFrame, start_cash: float = 10_000.0, contract_size: float = 1.0):
        self.df_raw = df_raw.reset_index(drop=True)
        self.data, self.meta = build_features(self.df_raw)
        self.i = 0
        self.pos = Position()
        self.cash = float(start_cash)
        self.equity_hist = [self.cash]
        self.contract = float(contract_size)
        self.daily_risk_used = 0.0
        self.cur_day = None
        # bookkeeping
        self._open_groups = 0  # สำหรับจำลอง group/pyramid อย่างง่าย

    def _price(self) -> float:
        return float(self.data.loc[self.i, "close"])

    def _bar_date(self):
        # รองรับทั้ง str / pandas.Timestamp
        t = self.data.loc[self.i, "time"]
        return pd.to_datetime(t).date()

    def _position_value(self, price: float) -> float:
        return self.pos.qty * (price - self.pos.entry) * self.contract

    def _risk_per_trade_value(self) -> float:
        return self.equity_hist[-1] * config.RISK_PER_TRADE

    def _daily_reset_if_needed(self):
        d = self._bar_date()
        if self.cur_day is None:
            self.cur_day = d
        elif d != self.cur_day:
            self.cur_day = d
            self.daily_risk_used = 0.0

    def reset(self) -> Dict[str, Any]:
        self.i = max(50, config.ATR_PERIOD + 1)  # ให้ indicator อุ่นตัว
        self.pos = Position()
        self.cash = float(self.equity_hist[0])
        self.equity_hist = [self.cash]
        self.daily_risk_used = 0.0
        self.cur_day = None
        self._open_groups = 0
        return self._obs()

    def _obs(self) -> Dict[str, Any]:
        row = self.data.iloc[self.i]
        return {
            "close": float(row["close"]),
            "ema_fast": float(row["ema_fast"]),
            "ema_slow": float(row["ema_slow"]),
            "ema_trend": float(row["ema_trend"]),
            "rsi14": float(row["rsi14"]) if not np.isnan(row["rsi14"]) else 50.0,
            "atr14": float(row["atr14"]) if not np.isnan(row["atr14"]) else 0.0,
            "is_spike": bool(row["is_spike"]),
            "trend_up": int(row["trend_up"]),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        action: 0 hold, 1 open/add long, 2 exit-all
        """
        assert action in (0, 1, 2)
        done = False
        info: Dict[str, Any] = {}
        self._daily_reset_if_needed()

        # ใช้ราคาปิดเท่านั้น (on close bar)
        px = self._price()
        prev_equity = self.equity_hist[-1]

        # Spike filter: skip ทุกอย่างถ้าเป็น spike bar
        if self.data.loc[self.i, "is_spike"]:
            action = 0  # บังคับ hold

        # Risk/day cap guard
        if self.daily_risk_used >= config.DAILY_RISK_CAP * prev_equity:
            action = 0

        # --- Execute ---
        step_pnl = 0.0
        if action == 1:
            # open/add long เฉพาะเมื่อ trend ok และ cross up (เป็น entry เชิงตัวอย่าง)
            can_long = (self.data.loc[self.i, "ema_fast"] > self.data.loc[self.i, "ema_slow"]) \
                       and (self.data.loc[self.i, "close"] > self.data.loc[self.i, "ema_trend"])
            if can_long:
                risk_val = self._risk_per_trade_value()
                # ตั้ง SL = entry - ATR*X เพื่อคำนวณปริมาณ
                atr = float(self.data.loc[self.i, "atr14"]) or 0.0
                sl_buffer = max(atr * 1.5, 0.5)  # กัน 0
                stop_px = px - sl_buffer
                per_unit_risk = max(px - stop_px, 1e-4) * self.contract
                qty = max(int(risk_val / per_unit_risk), 1)
                if self.pos.side == 0:
                    self.pos = Position(side=+1, entry=px, qty=qty, sl=stop_px, group_id=self._open_groups)
                elif self.pos.side == +1:
                    # pyramiding เฉพาะเมื่อกำไร
                    if px > self.pos.entry:
                        self.pos.qty += qty
                # บันทึก risk ใช้ไป (คง conservative เล็กน้อย)
                self.daily_risk_used += config.RISK_PER_TRADE * prev_equity

        elif action == 2:
            # exit all
            if self.pos.side != 0:
                step_pnl += self._position_value(px)
                self.cash += step_pnl
                self.pos = Position()  # flat

        # move trailing SL / breakeven guard (อย่างย่อ)
        if self.pos.side == +1 and self.pos.qty > 0:
            atr = float(self.data.loc[self.i, "atr14"]) or 0.0
            # breakeven guard หลังได้ 1R
            R = (px - self.pos.entry) / max(atr, 1e-4)
            if R >= config.BREAKEVEN_AFTER_R_MULT:
                self.pos.sl = max(self.pos.sl or self.pos.entry, self.pos.entry)
            # ATR trailing
            trail = px - config.ATR_TRAIL_MULT * atr
            if self.pos.sl is None:
                self.pos.sl = trail
            else:
                self.pos.sl = max(self.pos.sl, trail)
            # ถ้าหลุด SL → ปิดสถานะ
            if px <= (self.pos.sl or -1e9):
                step_pnl += self._position_value(self.pos.sl)
                self.cash += step_pnl
                self.pos = Position()

        # MTM equity
        mtm = self._position_value(px)
        equity = self.cash + mtm
        self.equity_hist.append(equity)

        # reward
        reward = step_reward(step_pnl=equity - prev_equity, equity_curve=np.array(self.equity_hist, dtype=float))

        # step next
        self.i += 1
        if self.i >= len(self.data) - 1:
            done = True
        return self._obs(), float(reward), bool(done), info
