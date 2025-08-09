# regime_detector.py
import numpy as np
import pandas as pd
from config import EMA_PERIODS, EMA_SPREAD_THRESHOLD

def detect_market_regime(df):
    ema_short, ema_mid, ema_long = EMA_PERIODS
    df["ema_s"] = df["close"].ewm(span=ema_short).mean()
    df["ema_m"] = df["close"].ewm(span=ema_mid).mean()
    df["ema_l"] = df["close"].ewm(span=ema_long).mean()

    spread = (df["ema_s"] - df["ema_l"]) / df["ema_l"]

    regime = np.where(spread > EMA_SPREAD_THRESHOLD, 1,  # Uptrend
             np.where(spread < -EMA_SPREAD_THRESHOLD, -1,  # Downtrend
             0))  # Sideways

    return regime[-1]  # latest regime
