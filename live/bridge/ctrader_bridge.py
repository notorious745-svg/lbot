# Stub bridge for cTrader. ใช้โหมด SIM จาก CSV ได้ทันที
import time
import pandas as pd
from typing import Callable, List, Any

CB = Callable[[dict], None]
_callbacks: List[CB] = []
_hist = None

def connect(): return True
def subscribe(symbol: str, timeframe: str): pass
def on_bar_close(cb: CB): _callbacks.append(cb)
def positions(symbol: str): return []  # TODO: ต่อ API จริง
def place(side: str, volume: float, sl: float=None, tp: float=None, label: str=""):
    print(f"[SIM] place {side} vol={volume:.2f} sl={sl} tp={tp} label={label}")
def modify_sl(position_id: Any, sl: float): print(f"[SIM] modify_sl {position_id} -> {sl}")
def close(position_id: Any): print(f"[SIM] close {position_id}")
def price(symbol: str) -> float:
    if _hist is not None and len(_hist): return float(_hist.iloc[-1]["close"])
    return float("nan")

def run_sim_from_csv(csv_path: str, sleep_sec: float=0.05, tail_window: int=300):
    """อ่าน CSV: time,open,high,low,close,volume แล้วจำลอง event bar-close"""
    global _hist
    df = pd.read_csv(csv_path)
    if "time" not in df.columns: raise ValueError("CSV mustมีคอลัมน์ time")
    df["time"] = pd.to_datetime(df["time"])
    for i in range(max(100, tail_window), len(df)):
        _hist = df.iloc[:i].copy()
        bar = {
            "time": df.iloc[i]["time"].to_pydatetime(),
            "open": float(df.iloc[i]["open"]),
            "high": float(df.iloc[i]["high"]),
            "low": float(df.iloc[i]["low"]),
            "close": float(df.iloc[i]["close"]),
            "volume": float(df.iloc[i]["volume"]),
            "history": _hist[["time","open","high","low","close","volume"]].reset_index(drop=True),
        }
        for cb in _callbacks: cb(bar)
        time.sleep(sleep_sec)
