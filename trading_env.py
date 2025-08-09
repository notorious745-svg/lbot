# trading_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import ta

from utils import load_data
from config import DATA_PATH, INITIAL_BALANCE, LOT_SIZE, MAX_STEPS

# ===== DEFAULT PARAMS (ถ้าไม่มีใน config.py ก็ใช้ค่าพวกนี้) =====
ATR_PERIOD        = 14
RISK_PCT_TRADE    = 0.005     # เสี่ยง 0.5% ของ equity ต่อไม้
ATR_SL_MULT       = 2.0       # ระยะ SL = k * ATR
ATR_TRAIL_MULT    = 1.0       # ระยะ trailing = k * ATR
BE_AFTER_ATR      = 0.5       # กำไรถึง k*ATR → ขยับ SL มา breakeven
UNITS_MIN         = 0.1
UNITS_MAX         = 3.0
SPIKE_ATR_MULT    = 1.2       # ถ้า |Δราคา| > k*ATR → ไม่เปิดไม้ใหม่
SPREAD_COST       = 0.08      # ค่าประมาณ spread (USD/oz) ต่อ round-trip ต่อ 1 unit
COMMISSION_PER_UNIT = 0.0     # ค่าคอมต่อ unit ต่อเทรด (ใส่ตามโบรกได้)

def _safe(val, default):
    try:
        return float(val)
    except Exception:
        return default


class TradingEnv(gym.Env):
    """
    Actions: 0=Hold, 1=Long, 2=Short
    Obs: ฟีเจอร์ตัวเลข (float32)
    PnL/Reward: mark-to-market รายบาร์ + shaping ตอนจบ episode ผ่านสถิติ
    Money Management: ขนาดไม้แปรตาม RISK_PCT_TRADE และ ATR
    Exit: Breakeven & ATR trailing, ตรวจ SL ทุกบาร์ (soft stop)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        raw = load_data(DATA_PATH)  # ต้องมี time, open, high, low, close, volume
        if not np.issubdtype(raw["time"].dtype, np.datetime64):
            raw["time"] = pd.to_datetime(raw["time"])

        # keep raw arrays for pricing/ATR/time
        self._time = raw["time"].to_numpy()
        self._close = raw["close"].to_numpy(dtype=float)
        self._high = raw["high"].to_numpy(dtype=float)
        self._low  = raw["low"].to_numpy(dtype=float)

        # raw ATR for risk sizing & stops
        raw_atr = ta.volatility.average_true_range(
            high=raw["high"], low=raw["low"], close=raw["close"], window=ATR_PERIOD
        )
        self._atr = raw_atr.to_numpy(dtype=float)

        # build features (numeric only) for observation
        feat = raw.copy()
        feat["ema10"]  = ta.trend.ema_indicator(feat["close"], window=10)
        feat["ema25"]  = ta.trend.ema_indicator(feat["close"], window=25)
        feat["ema50"]  = ta.trend.ema_indicator(feat["close"], window=50)
        feat["rsi14"]  = ta.momentum.rsi(feat["close"], window=14)
        feat["atr14"]  = raw_atr
        feat["ch20_h"] = feat["high"].rolling(20).max()
        feat["ch20_l"] = feat["low"].rolling(20).min()
        feat["ch55_h"] = feat["high"].rolling(55).max()
        feat["ch55_l"] = feat["low"].rolling(55).min()
        feat = feat.dropna().reset_index(drop=True)

        # align raw arrays to features length (ตัดหัว NaN ออก)
        cut = len(feat)
        self._close = self._close[-cut:]
        self._high  = self._high[-cut:]
        self._low   = self._low[-cut:]
        self._time  = self._time[-cut:]
        self._atr   = self._atr[-cut:]

        # normalize numeric features (ไม่แตะ time)
        num_cols = feat.select_dtypes(include=[np.number]).columns
        mu = feat[num_cols].mean()
        sd = feat[num_cols].std().replace(0, 1.0)
        feat[num_cols] = (feat[num_cols] - mu) / (sd + 1e-9)

        self.df = feat[num_cols].copy()
        self.feat_cols = list(self.df.columns)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feat_cols),), dtype=np.float32
        )

        # episode state
        self.current_step = 0
        self.step_count = 0
        self.balance = float(INITIAL_BALANCE)
        self.equity  = float(INITIAL_BALANCE)

        # position state
        self.pos_sign = 0      # -1 / 0 / +1
        self.pos_units = 0.0
        self.entry_price = 0.0
        self.sl = None
        self._current_trade_pnl = 0.0  # sum of mtm since entry (for metrics)

        # logs
        self.equity_curve = [self.equity]
        self.trade_pnls: list[float] = []
        self.trades = 0

        self.n_bars = len(self.df)

    # ===== Gymnasium API =====
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.step_count = 0
        self.balance = float(INITIAL_BALANCE)
        self.equity  = float(INITIAL_BALANCE)
        self.pos_sign = 0
        self.pos_units = 0.0
        self.entry_price = 0.0
        self.sl = None
        self._current_trade_pnl = 0.0
        self.equity_curve = [self.equity]
        self.trade_pnls = []
        self.trades = 0
        return self._obs(), {}

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0

        idx = self.current_step
        price = self._close[idx]
        atr   = max(self._atr[idx], 1e-9)

        # === 1) Update MTM & trailing/breakeven if in position ===
        if self.pos_sign != 0 and idx > 0:
            prev_price = self._close[idx - 1]
            pnl_delta = (price - prev_price) * (1 if self.pos_sign > 0 else -1) \
                        * self.pos_units * LOT_SIZE
            self.balance += pnl_delta
            self._current_trade_pnl += pnl_delta
            reward += pnl_delta / max(abs(self.balance), 1e-6)  # step shaping เบาๆ

            # Breakeven trigger
            if (self.pos_sign > 0 and price - self.entry_price >= BE_AFTER_ATR * atr) or \
               (self.pos_sign < 0 and self.entry_price - price >= BE_AFTER_ATR * atr):
                self.sl = max(self.sl, self.entry_price) if self.pos_sign > 0 else \
                          min(self.sl, self.entry_price)

            # ATR trailing
            if self.pos_sign > 0:
                trail = price - ATR_TRAIL_MULT * atr
                self.sl = max(self.sl or -np.inf, trail)
            elif self.pos_sign < 0:
                trail = price + ATR_TRAIL_MULT * atr
                self.sl = min(self.sl or np.inf, trail)

            # Check stop
            if (self.pos_sign > 0 and price <= (self.sl or -np.inf)) or \
               (self.pos_sign < 0 and price >= (self.sl or np.inf)):
                self._close_trade(price)

        # === 2) Handle action / open or flip position ===
        if action in (1, 2):
            target_sign = 1 if action == 1 else -1

            # spike filter: ห้ามเปิดไม้ใหม่เมื่อเหวี่ยงแรง
            if idx > 0:
                if abs(self._close[idx] - self._close[idx - 1]) > SPIKE_ATR_MULT * atr:
                    target_sign = 0  # ignore signal

            if target_sign != self.pos_sign:
                # flip: close old first
                if self.pos_sign != 0:
                    self._close_trade(price)
                # open new
                if target_sign != 0:
                    self._open_trade(target_sign, price, atr)

        # === 3) advance ===
        self.equity = self.balance
        self.equity_curve.append(self.equity)

        self.current_step += 1
        self.step_count += 1

        if self.current_step >= self.n_bars - 1:
            terminated = True
        if self.step_count >= int(MAX_STEPS):
            truncated = True

        if terminated or truncated:
            if self.pos_sign != 0:
                self._close_trade(self._close[min(self.current_step, self.n_bars - 1)])

        return self._obs(), float(reward), terminated, truncated, {}

    # ===== helpers =====
    def _obs(self):
        return self.df.loc[self.current_step, self.feat_cols].to_numpy(dtype=np.float32)

    def _open_trade(self, sign: int, price: float, atr: float):
        # risk-based sizing: risk = equity * RISK_PCT_TRADE, SL distance = ATR_SL_MULT*ATR
        risk_cash = self.equity * RISK_PCT_TRADE
        sl_dist = ATR_SL_MULT * atr
        units = risk_cash / max(sl_dist * LOT_SIZE, 1e-9)
        units = float(np.clip(units, UNITS_MIN, UNITS_MAX))

        self.pos_sign = sign
        self.pos_units = units
        self.entry_price = price
        self.sl = price - sl_dist if sign > 0 else price + sl_dist
        self._current_trade_pnl = 0.0
        self.trades += 1  # count immediately on open

    def _close_trade(self, exit_price: float):
        # realize costs only here (MTM ถูกสะสมไปแล้ว)
        round_cost = (SPREAD_COST * self.pos_units * LOT_SIZE) + \
                     (COMMISSION_PER_UNIT * self.pos_units)
        self.balance -= round_cost

        # สำหรับสถิติระดับ "ไม้": เอา PnL ตั้งแต่เปิดจนปิด หักต้นทุน
        realized = self._current_trade_pnl - round_cost
        self.trade_pnls.append(realized)

        # reset position
        self.pos_sign = 0
        self.pos_units = 0.0
        self.entry_price = 0.0
        self.sl = None
        self._current_trade_pnl = 0.0

    def render(self):
        print(f"t={self.current_step} pos={self.pos_sign} units={self.pos_units:.2f} "
              f"bal={self.balance:.2f} eq={self.equity:.2f}")
