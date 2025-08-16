from __future__ import annotations
from pathlib import Path
from datetime import timezone
import pandas as pd
from config import DATA_FILE, LOCAL_TZ

REQUIRED_COLS = ["time", "open", "high", "low", "close", "volume"]

def _to_bkk(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        t = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        t = series
        if getattr(t.dt, "tz", None) is None:
            t = t.dt.tz_localize(timezone.utc)
    return t.dt.tz_convert(LOCAL_TZ)

def load_price_csv(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path or DATA_FILE)
    df = pd.read_csv(csv_path)
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss} :: {csv_path}")
    df["time"] = _to_bkk(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["range"] = (df["high"] - df["low"]).abs()
    return df
