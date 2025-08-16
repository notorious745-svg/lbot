from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import DATA_FILE, LOCAL_TZ

REQUIRED_COLS = ["time", "open", "high", "low", "close", "volume"]

def _ensure_bkk(dt_series: pd.Series) -> pd.Series:
    # รับได้ทั้ง string/naive/aware → คืนค่าเป็น Asia/Bangkok ที่ปลอดภัย
    if not pd.api.types.is_datetime64_any_dtype(dt_series):
        t = pd.to_datetime(dt_series, errors="coerce", utc=True)
    else:
        t = dt_series
        if getattr(t.dt, "tz", None) is None:
            t = t.dt.tz_localize("UTC")
    return t.dt.tz_convert(LOCAL_TZ)

def _make_demo_data(n_days: int = 14, start_close: float = 2400.0, mu: float = 0.0, sigma: float = 0.008) -> pd.DataFrame:
    """
    สร้างเดโมดาต้า M15 ต่อเนื่อง ~14 วันย้อนหลัง เพื่อให้ CI/backtest วิ่งได้แม้ไม่มี CSV จริง
    ใช้ GBM แบบง่าย + สุ่มสร้าง OHLC จาก close
    """
    end = datetime.now(tz=LOCAL_TZ).replace(second=0, microsecond=0)
    start = end - timedelta(days=n_days)
    # ทำกริดเวลา M15
    idx = pd.date_range(start=start, end=end, freq="15min", tz=LOCAL_TZ)
    n = len(idx)
    # สร้าง close ด้วย GBM
    rng = np.random.default_rng(1234)
    rets = rng.normal(loc=mu / (96), scale=sigma / np.sqrt(96), size=n)  # 96 แท่ง/วัน
    close = np.empty(n)
    close[0] = start_close
    for i in range(1, n):
        close[i] = close[i-1] * (1.0 + rets[i])
    # สร้าง OHLC คร่าว ๆ รอบ close
    spread = np.maximum(close * 0.0008, 0.1)  # ~8 bps
    open_  = np.roll(close, 1); open_[0] = close[0]
    high   = np.maximum(close, open_) + spread * rng.random(n)
    low    = np.minimum(close, open_) - spread * rng.random(n)
    vol    = rng.integers(50, 300, size=n)

    df = pd.DataFrame({
        "time": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })
    df["range"] = (df["high"] - df["low"]).abs()
    return df

def load_price_csv(path: Path | None = None) -> pd.DataFrame:
    """
    พยายามอ่าน CSV จริงตาม DATA_FILE; ถ้าไม่พบ ให้สร้างเดโมดาต้าอัตโนมัติ (กัน CI ล้ม)
    """
    csv_path = Path(path or DATA_FILE)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing} :: {csv_path}")
        df["time"] = _ensure_bkk(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["range"] = (df["high"] - df["low"]).abs()
        return df

    # Fallback: demo data
    df = _make_demo_data()
    return df
