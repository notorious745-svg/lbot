# core/runner.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

import config
from core.entries import decide_entry
from core.position_manager import PMConfig, PositionManager

def load_data(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"DATA not found: {f!s}")
    df = pd.read_csv(f)
    # columns: time,open,high,low,close,volume
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=50, adjust=False).mean()
    ema_slow = df["close"].ewm(span=200, adjust=False).mean()
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    atr = tr.ewm(span=config.ATR_PERIOD, adjust=False).mean()
    df = df.assign(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        atr=atr,
        bar_closed=True  # backtest ปกติใช้ตอนแท่งปิด
    )
    return df.dropna().reset_index(drop=True)

def backtest(df: pd.DataFrame):
    pm = PositionManager(
        PMConfig(
            risk_per_trade=config.RISK_PER_TRADE,
            daily_risk_cap=config.DAILY_RISK_CAP,
            max_pyramid=config.MAX_PYRAMID,
            add_on_gain_r=config.ADD_ON_GAIN_R_MULT,
            breakeven_after_r=config.BREAKEVEN_AFTER_R_MULT,
            atr_mult=config.ATR_TRAIL_MULT,
            pip=0.01,
        ),
        equity=config.START_BALANCE,
    )
    balance = config.START_BALANCE
    equity_curve = []

    last_date = None
    daily_risk_reset = False

    for i, row in df.iterrows():
        # reset daily risk
        d = row["time"].date()
        if last_date is None or d != last_date:
            pm.day_risk_used = 0.0
            daily_risk_reset = True
        last_date = d

        price = float(row["close"])
        atr = float(row["atr"])
        low = float(row["low"])

        # 1) exit by SL
        if pm.positions and pm.stop_out(low):
            balance += pm.exit_all(price)

        # 2) manage trailing/breakeven
        if pm.positions:
            pm.update_sl(price, atr)

        # 3) decide entry
        sig = decide_entry(
            {
                "open": float(row["open"]),
                "close": price,
                "ema_fast": float(row["ema_fast"]),
                "ema_slow": float(row["ema_slow"]),
                "atr": atr,
                "bar_closed": True,
            },
            bar_close_only=config.BAR_CLOSE_ONLY,
            use_spike=config.USE_SPIKE_FILTER,
            spike_k=config.SPIKE_MULT,
        )
        if sig == 1:
            pm.try_open_or_add(price, atr)

        # mark-to-market
        eq = balance + pm.unrealized(price)
        equity_curve.append(eq)

    # ปิดสุดท้ายถ้ายังมี position
    if pm.positions:
        balance += pm.exit_all(df.iloc[-1]["close"])

    equity_curve[-1] = balance
    return balance, pd.Series(equity_curve, index=df["time"])

def main():
    df = load_data(config.DATA_FILE)
    df = add_indicators(df)
    final_bal, curve = backtest(df)

    ret = curve.pct_change().fillna(0.0)
    sharpe = 0.0 if ret.std() == 0 else (ret.mean() / ret.std()) * (np.sqrt(365*24*4))  # scale ~M15
    print(f"Equity final: {final_bal:,.2f} | Sharpe~ {sharpe:.2f} | Steps: {len(curve):,}")

if __name__ == "__main__":
    main()
