# features.py
import pandas as pd
import numpy as np
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA ต่าง ๆ
    df["ema10"]  = ta.trend.ema_indicator(df["close"], window=10)
    df["ema25"]  = ta.trend.ema_indicator(df["close"], window=25)
    df["ema50"]  = ta.trend.ema_indicator(df["close"], window=50)
    df["ema100"] = ta.trend.ema_indicator(df["close"], window=100)  # <== เพิ่ม

    # Momentum / RSI / ROC
    df["rsi"]    = ta.momentum.rsi(df["close"], window=14)
    df["roc10"]  = ta.momentum.roc(df["close"], window=10)          # <== เพิ่ม

    # ATR
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    # Turtle channel
    df["turtle_high20"] = df["high"].rolling(20).max()
    df["turtle_low20"]  = df["low"].rolling(20).min()
    df["turtle_high55"] = df["high"].rolling(55).max()
    df["turtle_low55"]  = df["low"].rolling(55).min()

    # จัด index ให้สะอาด
    df = df.dropna().reset_index(drop=True)
    return df

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df
    mean = df[num_cols].mean()
    std  = df[num_cols].std().replace(0, 1.0)
    df[num_cols] = (df[num_cols] - mean) / (std + 1e-9)
    return df

def build_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df = normalize_features(df)
    # ใช้เฉพาะคอลัมน์ตัวเลข (OHLCV + อินดิเคเตอร์) -> ควรได้ 16 คอลัมน์
    num = df.select_dtypes(include=[np.number]).columns
    return df[num]
