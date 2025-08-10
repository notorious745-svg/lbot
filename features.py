# lbot/features.py
import numpy as np
import pandas as pd

def build_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    รับ df ที่มีคอลัมน์อย่างน้อย: time, open, high, low, close, volume
    คืน DataFrame ฟีเจอร์ที่ 'คงรูป' ใช้ได้ทั้งตอนเทรนและตอน live
    """
    df = df.copy()

    # ensure base cols
    base = ["open","high","low","close","volume"]
    for c in base:
        if c not in df.columns:
            df[c] = np.nan

    # returns & ranges
    df["ret_1"]   = df["close"].pct_change(1).fillna(0.0)
    df["ret_3"]   = df["close"].pct_change(3).fillna(0.0)
    df["ret_6"]   = df["close"].pct_change(6).fillna(0.0)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    # rolling stats
    def _roll(z, w): return z.rolling(w, min_periods=1)
    df["vol_10"]  = _roll(df["ret_1"], 10).std().fillna(0.0)
    df["vol_30"]  = _roll(df["ret_1"], 30).std().fillna(0.0)
    df["ma_10"]   = _roll(df["close"], 10).mean().fillna(method="bfill")
    df["ma_30"]   = _roll(df["close"], 30).mean().fillna(method="bfill")
    df["ma_gap"]  = (df["ma_10"] - df["ma_30"]) / df["ma_30"].replace(0, np.nan)

    # volume features
    df["v_ma_10"] = _roll(df["volume"], 10).mean().replace(0, np.nan)
    df["v_rel"]   = df["volume"] / df["v_ma_10"]
    df["v_rel"]   = df["v_rel"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
    feats = df.select_dtypes(include=[np.number]).copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feats
