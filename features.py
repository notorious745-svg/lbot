from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import config

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return arr
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[:] = np.nan
    prev = np.nan
    for i, v in enumerate(arr):
        if np.isnan(v):
            out[i] = np.nan
            continue
        if np.isnan(prev):
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    def rma(x, n):
        out = np.empty_like(x, dtype=float)
        out[:] = np.nan
        acc = 0.0
        cnt = 0
        alpha = 1.0 / n
        for i, v in enumerate(x):
            if np.isnan(v):
                out[i] = np.nan
                continue
            if cnt < n:
                acc += v
                cnt += 1
                out[i] = np.nan if cnt < n else acc / n
            else:
                acc = (1 - alpha) * out[i-1] + alpha * v
                out[i] = acc
        return out
    avg_up = rma(up, period)
    avg_dn = rma(dn, period)
    rs = avg_up / (avg_dn + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    # RMA/TR EMA แบบ alpha=1/period
    alpha = 1.0 / period
    out = np.empty_like(tr, dtype=float)
    out[:] = np.nan
    acc = 0.0
    cnt = 0
    for i, v in enumerate(tr):
        if np.isnan(v):
            out[i] = np.nan
            continue
        if cnt < period:
            acc += v
            cnt += 1
            out[i] = np.nan if cnt < period else acc / period
        else:
            out[i] = (1 - alpha) * out[i-1] + alpha * v
    return out

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Expect df columns: ['time','open','high','low','close','volume']
    Returns: features_df with added columns + meta (for runner/env)
    """
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)

    df["ema_fast"] = _ema(c, config.EMA_FAST)
    df["ema_slow"] = _ema(c, config.EMA_SLOW)
    df["ema_trend"] = _ema(c, config.EMA_TREND)
    df["rsi14"] = _rsi(c, 14)
    df["atr14"] = _atr(h, l, c, config.ATR_PERIOD)

    # spike filter helper
    df["bar_range"] = (df["high"] - df["low"]).astype(float)
    df["is_spike"] = (df["bar_range"] > (config.SPIKE_ATR_MULT * df["atr14"]).fillna(np.inf))

    # convenience trend flags
    df["trend_up"] = (df["ema_trend"] < df["close"]).astype(int)
    df["ema_cross_up"] = ((df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))).astype(int)
    df["ema_cross_dn"] = ((df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))).astype(int)

    meta = {
        "symbol": config.SYMBOL,
        "tf_m": str(config.TIMEFRAME_MINUTES),
        "bar_close_only": str(config.BAR_CLOSE_ONLY),
    }
    return df, meta
