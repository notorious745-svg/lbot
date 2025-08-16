# core/position_manager.py
from __future__ import annotations
import numpy as np
import pandas as pd
from config import MAX_POS_TOTAL
from core.indicators import atr

# สัญญาณมาตรฐานที่ใช้รวมเป็นฐาน
STRATS = ["ema", "turtle20", "turtle55", "meanrev"]

def base_target_from_signals(sig: pd.DataFrame) -> pd.Series:
    cols = [c for c in STRATS if c in sig.columns]
    if not cols:
        raise ValueError("No known strategy signal columns in DataFrame.")
    base = sig[cols].clip(-1, 1).sum(axis=1)
    base = base.clip(-MAX_POS_TOTAL, MAX_POS_TOTAL)
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
    """
    สร้างซีรีส์ตำแหน่งถือครอง (pos) โดยคำนึงถึง:
      - base target จากสัญญาณรวม
      - pyramiding เฉพาะกำไร (แยกชั้นตาม ATR)
      - ATR trailing stop จาก MFE
    pos เป็นจำนวนหน่วยรวม (-MAX_POS_TOTAL..MAX_POS_TOTAL) ต่อบาร์
    """
    idx = df.index
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)

    base = base_target_from_signals(sig).reindex(idx)
    atr_s = atr(df, n=atr_n).reindex(idx)
    # กัน NaN ATR
    if atr_s.isna().all():
        atr_s = pd.Series(1e-6, index=idx)
    else:
        fill = np.nanmedian(atr_s.dropna().values) if atr_s.notna().any() else 1e-6
        atr_s = atr_s.fillna(method="ffill").fillna(method="bfill").fillna(fill)

    pos = np.zeros(n, dtype=float)

    # state
    cur_pos = 0
    avg_entry = np.nan
    layers_used = 0
    # trailing
    peak = None    # สำหรับ long
    trough = None  # สำหรับ short
    trail = None

    close = df["close"].values

    for i in range(n):
        price = close[i]
        a = max(1e-9, float(atr_s.iat[i]))
        bt = int(base.iat[i])

        # ปรับ trailing ตาม MFE
        if cur_pos > 0:
            peak = price if (peak is None) else max(peak, price)
            # trail candidate: peak - k*ATR
            t_candidate = peak - atr_mult * a
            trail = t_candidate if (trail is None) else max(trail, t_candidate)
            # ตัดขาดทุน/ยอมรับกำไร
            if price <= trail:
                cur_pos = 0
                avg_entry = np.nan
                layers_used = 0
                peak = trough = trail = None

        elif cur_pos < 0:
            trough = price if (trough is None) else min(trough, price)
            t_candidate = trough + atr_mult * a
            trail = t_candidate if (trail is None) else min(trail, t_candidate)
            if price >= trail:
                cur_pos = 0
                avg_entry = np.nan
                layers_used = 0
                peak = trough = trail = None

        # ถ้า opposite ชัดเจน → ปิดก่อน
        if flatten_on_opposite and cur_pos != 0 and (bt * cur_pos < 0):
            cur_pos = 0
            avg_entry = np.nan
            layers_used = 0
            peak = trough = trail = None

        # เปิดไม้แรกหรือเพิ่มชั้นเฉพาะกำไร
        if cur_pos == 0:
            if bt > 0:
                cur_pos = 1
                avg_entry = price
                layers_used = 0
                peak = price; trough = None
                trail = peak - atr_mult * a
            elif bt < 0:
                cur_pos = -1
                avg_entry = price
                layers_used = 0
                trough = price; peak = None
                trail = trough + atr_mult * a

        else:
            # Pyramiding เฉพาะกำไร
            if cur_pos > 0 and bt > cur_pos and layers_used < max_layers:
                trigger = avg_entry + (layers_used + 1) * pyramid_step_atr * a
                if price >= trigger:
                    add = min(1, MAX_POS_TOTAL - cur_pos)
                    if add > 0:
                        avg_entry = (avg_entry * cur_pos + price * add) / (cur_pos + add)
                        cur_pos += add
                        layers_used += 1
                        peak = price if peak is None else max(peak, price)
                        trail = max(trail, peak - atr_mult * a) if trail is not None else peak - atr_mult * a

            elif cur_pos < 0 and bt < cur_pos and layers_used < max_layers:
                trigger = avg_entry - (layers_used + 1) * pyramid_step_atr * a
                if price <= trigger:
                    add = min(1, MAX_POS_TOTAL - abs(cur_pos))
                    if add > 0:
                        # สำหรับ short ใช้ค่าเฉลี่ยถ่วงน้ำหนักของราคาเข้า
                        avg_entry = (avg_entry * abs(cur_pos) + price * add) / (abs(cur_pos) + add)
                        cur_pos -= add
                        layers_used += 1
                        trough = price if trough is None else min(trough, price)
                        trail = min(trail, trough + atr_mult * a) if trail is not None else trough + atr_mult * a

        # รับประกันไม่เกิน MAX_POS_TOTAL
        cur_pos = max(-MAX_POS_TOTAL, min(MAX_POS_TOTAL, cur_pos))
        pos[i] = cur_pos

    return pd.Series(pos, index=idx, name="pos")
