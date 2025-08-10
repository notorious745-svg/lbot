# features.py
import pandas as pd
import numpy as np
import ta

# ลิสต์คอลัมน์ให้คงที่ (ป้องกันสลับลำดับ)
COLS = [
    "open", "high", "low", "close", "volume",
    "ema10", "ema25", "ema50", "ema100",
    "rsi", "roc10", "atr",
    "turtle_high20", "turtle_low20", "turtle_high55", "turtle_low55",
]

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema10"]  = ta.trend.ema_indicator(df["close"], window=10)
    df["ema25"]  = ta.trend.ema_indicator(df["close"], window=25)
    df["ema50"]  = ta.trend.ema_indicator(df["close"], window=50)
    df["ema100"] = ta.trend.ema_indicator(df["close"], window=100)

    # Momentum
    df["rsi"]   = ta.momentum.rsi(df["close"], window=14)
    df["roc10"] = ta.momentum.roc(df["close"], window=10)

    # Volatility
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    # Turtle channels
    df["turtle_high20"] = df["high"].rolling(20).max()
    df["turtle_low20"]  = df["low"].rolling(20).min()
    df["turtle_high55"] = df["high"].rolling(55).max()
    df["turtle_low55"]  = df["low"].rolling(55).min()

    # จัดการ NaN จากอินดิเคเตอร์ช่วงต้น ๆ
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

    # ให้แน่ใจว่ามีครบ 16 ฟีเจอร์ตามลำดับที่กำหนด
    # (ถ้าข้อมูลต้นทางไม่มีบางคอลัมน์ ให้เติม 0 ไว้)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[COLS]
