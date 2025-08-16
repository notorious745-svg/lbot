from __future__ import annotations
import numpy as np
import pandas as pd
from config import MAX_POS_TOTAL
from core.indicators import atr

STRATS = ["ema", "turtle20", "turtle55", "meanrev"]

def base_target_from_signals(sig: pd.DataFrame) -> pd.Series:
    cols = [c for c in STRATS if c in sig.columns]
    if not cols:
        raise ValueError("No known strategy signal columns in DataFrame.")
    base = sig[cols].clip(-1, 1).sum(axis=1).clip(-MAX_POS_TOTAL, MAX_POS_TOTAL)
    return base.astype(int)

def generate_position_series(
    df: pd.DataFrame,
    sig: pd.DataFrame,
    atr_n: int = 14,
    atr_mult: float = 2.5,
    pyramid_step_atr: float = 1.0,
    max_layers: int = 2,
    flatten_on_opposite: bool = True,
) -> pd.Series:
    """รวมสัญญาณ -> base target + pyramiding เฉพาะกำไร + ATR trailing"""
    idx = df.index; n = len(df)
    if n == 0: return pd.Series(dtype=float)

    base = base_target_from_signals(sig).reindex(idx)
    a = atr(df, n=atr_n).reindex(idx)
    fill = np.nanmedian(a.dropna()) if a.notna().any() else 1e-6
    a = a.fillna(method="ffill").fillna(method="bfill").fillna(fill)

    pos = np.zeros(n, dtype=float)
    cur_pos = 0
    avg_entry = np.nan
    layers = 0
    peak = trough = trail = None
    px = df["close"].values

    for i in range(n):
        price = px[i]; atr_i = float(max(1e-9, a.iat[i])); bt = int(base.iat[i])

        # trailing
        if cur_pos > 0:
            peak = price if peak is None else max(peak, price)
            trail = max(trail or (peak - atr_mult*atr_i), peak - atr_mult*atr_i)
            if price <= trail:  # ปิด long
                cur_pos = 0; avg_entry = np.nan; layers = 0; peak = trough = trail = None

        elif cur_pos < 0:
            trough = price if trough is None else min(trough, price)
            trail = min(trail or (trough + atr_mult*atr_i), trough + atr_mult*atr_i)
            if price >= trail:  # ปิด short
                cur_pos = 0; avg_entry = np.nan; layers = 0; peak = trough = trail = None

        # opposite -> flatten ก่อน
        if flatten_on_opposite and cur_pos*bt < 0:
            cur_pos = 0; avg_entry = np.nan; layers = 0; peak = trough = trail = None

        # เปิดไม้แรก หรือ เพิ่มชั้นเฉพาะกำไร
        if cur_pos == 0:
            if bt > 0:
                cur_pos = 1; avg_entry = price; layers = 0
                peak = price; trough = None; trail = peak - atr_mult*atr_i
            elif bt < 0:
                cur_pos = -1; avg_entry = price; layers = 0
                trough = price; peak = None; trail = trough + atr_mult*atr_i
        else:
            if cur_pos > 0 and bt > cur_pos and layers < max_layers:
                trigger = avg_entry + (layers+1)*pyramid_step_atr*atr_i
                if price >= trigger:
                    add = min(1, MAX_POS_TOTAL - cur_pos)
                    if add > 0:
                        avg_entry = (avg_entry*cur_pos + price*add)/(cur_pos+add)
                        cur_pos += add; layers += 1
                        peak = max(peak or price, price); trail = max(trail or (peak - atr_mult*atr_i), peak - atr_mult*atr_i)
            elif cur_pos < 0 and bt < cur_pos and layers < max_layers:
                trigger = avg_entry - (layers+1)*pyramid_step_atr*atr_i
                if price <= trigger:
                    add = min(1, MAX_POS_TOTAL - abs(cur_pos))
                    if add > 0:
                        avg_entry = (avg_entry*abs(cur_pos) + price*add)/(abs(cur_pos)+add)
                        cur_pos -= add; layers += 1
                        trough = min(trough or price, price); trail = min(trail or (trough + atr_mult*atr_i), trough + atr_mult*atr_i)

        cur_pos = max(-MAX_POS_TOTAL, min(MAX_POS_TOTAL, cur_pos))
        pos[i] = cur_pos

    return pd.Series(pos, index=idx, name="pos")
