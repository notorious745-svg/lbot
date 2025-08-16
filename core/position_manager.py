from __future__ import annotations
import pandas as pd
from config import MAX_POS_TOTAL, MAX_POS_PER_STRAT

STRATS = ["ema", "turtle20", "turtle55", "meanrev"]

def apply_caps(sig: pd.DataFrame) -> pd.DataFrame:
    pos = sig[STRATS].clip(-1, 1).copy()
    pos["sum"] = pos.sum(axis=1).clip(-MAX_POS_TOTAL, MAX_POS_TOTAL)
    return pos

def build_orders(sig: pd.DataFrame) -> pd.DataFrame:
    pos = apply_caps(sig)
    turns = pos[STRATS].diff().fillna(0)
    orders = (turns != 0).astype(int)
    orders.columns = [f"order_{c}" for c in orders.columns]
    return pd.concat([pos, orders], axis=1)
