from __future__ import annotations
import os
import pandas as pd
import numpy as np
from datetime import datetime
import config
from features import build_features
from entries import decide_entry
from position_manager import PositionManager

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA not found: {path}")
    df = pd.read_csv(path)
    # คาดหวังคอลัมน์: time, open, high, low, close, volume
    df["time"] = pd.to_datetime(df["time"])
    return df

def run_backtest_simple():
    df_raw = load_data(config.DATA_FILE)
    df, meta = build_features(df_raw)

    pm = PositionManager(contract_size=1.0)
    cash = 10_000.0
    equity_hist = [cash]
    day = None

    # เริ่มหลังอินดิเคเตอร์อุ่นตัว
    start = max(50, config.ATR_PERIOD + 1)
    for i in range(start, len(df)):
        row = df.iloc[i]
        day_now = row["time"].date()
        if day != day_now:
            day = day_now
            pm.new_day(equity_hist[-1], day)

        # on-close logic
        obs = {
            "close": float(row["close"]),
            "ema_fast": float(row["ema_fast"]),
            "ema_slow": float(row["ema_slow"]),
            "ema_trend": float(row["ema_trend"]),
            "rsi14": float(row["rsi14"]) if not np.isnan(row["rsi14"]) else 50.0,
            "atr14": float(row["atr14"]) if not np.isnan(row["atr14"]) else 0.0,
            "is_spike": bool(row["is_spike"]),
        }

        action = 0
        if obs["is_spike"]:
            action = 0
        else:
            action = decide_entry(obs, bar_close_only=config.BAR_CLOSE_ONLY)
            # daily risk cap guard
            if pm.daily_risk_used >= config.DAILY_RISK_CAP * equity_hist[-1]:
                action = 0

        # execute
        step_pnl = 0.0
        if action == 1:
            if pm.can_add_long(obs["close"]):
                pm.open_or_add_long(obs["close"], obs["atr14"], equity_hist[-1])
        elif action == 2:
            step_pnl += pm.flat(obs["close"])

        # trailing & breakeven (ย่อ) — รักษาให้สอดคล้องกับ env
        if pm.state.side == +1 and pm.state.qty > 0:
            atr = obs["atr14"]
            # breakeven after 1R
            R = (obs["close"] - pm.state.entry) / max(atr, 1e-4)
            if R >= config.BREAKEVEN_AFTER_R_MULT:
                pm.state.sl = max(pm.state.sl or pm.state.entry, pm.state.entry)
            trail = obs["close"] - config.ATR_TRAIL_MULT * atr
            pm.state.sl = trail if pm.state.sl is None else max(pm.state.sl, trail)
            if obs["close"] <= (pm.state.sl or -1e9):
                step_pnl += pm.flat(pm.state.sl)

        mtm = pm.mtm(obs["close"])
        equity = cash + step_pnl + mtm
        cash += step_pnl
        equity_hist.append(equity)

    # สรุปผลคร่าว ๆ
    eq = np.array(equity_hist, dtype=float)
    ret = np.diff(eq) / np.maximum(eq[:-1], 1e-6)
    sharpe = (ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252*24*4)  # ประมาณคร่าว ๆ สำหรับ M15
    print(f"Equity final: {eq[-1]:.2f} | Sharpe~ {sharpe:.2f} | Steps: {len(eq)}")

if __name__ == "__main__":
    run_backtest_simple()
