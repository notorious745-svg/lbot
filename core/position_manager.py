from __future__ import annotations
import numpy as np
import pandas as pd
from config import MAX_POS_TOTAL
from core.indicators import atr

# สัญญาณที่รองรับ (ชื่อคอลัมน์จาก combined_signal)
STRATS = ["ema", "turtle20", "turtle55", "meanrev"]

def base_target_from_signals(sig: pd.DataFrame, vote_required: int = 2) -> pd.Series:
    """
    รวมสัญญาณเป็นค่าเป้าหมายพื้นฐานแบบ 'เสียงข้างมาก'
    vote_required=2 หมายถึง ต้องได้คะแนนรวม >=2 ถึงจะ long (<=-2 ถึงจะ short) มิฉะนั้น = 0
    """
    cols = [c for c in STRATS if c in sig.columns]
    if not cols:
        raise ValueError("No known strategy signal columns in DataFrame.")
    score = sig[cols].clip(-1, 1).sum(axis=1)
    bt = score.where(score.abs() >= vote_required, 0.0).apply(np.sign)
    return bt.clip(-MAX_POS_TOTAL, MAX_POS_TOTAL).astype(int)

def generate_position_series(
    df: pd.DataFrame,
    sig: pd.DataFrame,
    atr_n: int = 14,
    atr_mult: float = 3.0,
    pyramid_step_atr: float = 1.2,
    max_layers: int = 0,
    flatten_on_opposite: bool = True,
    vote_required: int = 2,          # ← ใหม่: ต้องได้เสียง >=2 ถึงจะเข้าทิศ
    cooldown_bars: int = 8,          # ← ใหม่: คูลดาวน์ก่อนเปิดไม้ใหม่ (8 แท่ง ~ 2 ชั่วโมงบน 15m)
) -> pd.Series:
    """
    รวมสัญญาณ -> base target (majority vote) + คูลดาวน์ + (option) pyramiding เฉพาะกำไร + ATR trailing
    """
    idx = df.index
    n = len(idx)
    if n == 0:
        return pd.Series(dtype=float)

    base = base_target_from_signals(sig, vote_required=vote_required).reindex(idx)
    a = atr(df, n=atr_n).reindex(idx)
    fill = np.nanmedian(a.dropna()) if a.notna().any() else 1e-6
    a = a.fillna(method="ffill").fillna(method="bfill").fillna(fill)

    pos = np.zeros(n, dtype=float)
    close = df["close"].values

    # state
    cur_pos = 0
    avg_entry = np.nan
    layers_used = 0
    peak = trough = trail = None
    last_change_i = -10**9  # ใช้กับ cooldown

    for i in range(n):
        price = float(close[i])
        atr_i = float(max(1e-9, a.iat[i]))
        bt = int(base.iat[i])

        # 1) trailing stop
        if cur_pos > 0:
            peak = price if peak is None else max(peak, price)
            trail_candidate = peak - atr_mult * atr_i
            trail = trail_candidate if trail is None else max(trail, trail_candidate)
            if price <= trail:
                cur_pos = 0; avg_entry = np.nan; layers_used = 0
                peak = trough = trail = None
                last_change_i = i  # เริ่มคูลดาวน์หลังปิด
        elif cur_pos < 0:
            trough = price if trough is None else min(trough, price)
            trail_candidate = trough + atr_mult * atr_i
            trail = trail_candidate if trail is None else min(trail, trail_candidate)
            if price >= trail:
                cur_pos = 0; avg_entry = np.nan; layers_used = 0
                peak = trough = trail = None
                last_change_i = i

        # 2) เจอสัญญาณตรงข้ามแรง → ปิดก่อน (ห้ามรีเวิร์สทันทีเพราะ cooldown จะกัน)
        if flatten_on_opposite and cur_pos != 0 and (bt * cur_pos < 0):
            cur_pos = 0; avg_entry = np.nan; layers_used = 0
            peak = trough = trail = None
            last_change_i = i

        # 3) เปิดไม้แรก (รอให้พ้นคูลดาวน์) หรือเพิ่มชั้นเฉพาะกำไร
        can_open = (i - last_change_i) >= cooldown_bars

        if cur_pos == 0:
            if can_open:
                if bt > 0:
                    cur_pos = 1; avg_entry = price; layers_used = 0
                    peak = price; trough = None; trail = peak - atr_mult * atr_i
                    last_change_i = i
                elif bt < 0:
                    cur_pos = -1; avg_entry = price; layers_used = 0
                    trough = price; peak = None; trail = trough + atr_mult * atr_i
                    last_change_i = i
        else:
            # Pyramiding (เฉพาะกำไร) — ไม่บังคับคูลดาวน์
            if max_layers > 0:
                if cur_pos > 0 and bt > cur_pos and layers_used < max_layers:
                    trigger = avg_entry + (layers_used + 1) * pyramid_step_atr * atr_i
                    if price >= trigger:
                        add = min(1, MAX_POS_TOTAL - cur_pos)
                        if add > 0:
                            avg_entry = (avg_entry * cur_pos + price * add) / (cur_pos + add)
                            cur_pos += add; layers_used += 1
                            peak = price if peak is None else max(peak, price)
                            trail = max(trail or (peak - atr_mult * atr_i), peak - atr_mult * atr_i)
                elif cur_pos < 0 and bt < cur_pos and layers_used < max_layers:
                    trigger = avg_entry - (layers_used + 1) * pyramid_step_atr * atr_i
                    if price <= trigger:
                        add = min(1, MAX_POS_TOTAL - abs(cur_pos))
                        if add > 0:
                            avg_entry = (avg_entry * abs(cur_pos) + price * add) / (abs(cur_pos) + add)
                            cur_pos -= add; layers_used += 1
                            trough = price if trough is None else min(trough, price)
                            trail = min(trail or (trough + atr_mult * atr_i), trough + atr_mult * atr_i)

        cur_pos = max(-MAX_POS_TOTAL, min(MAX_POS_TOTAL, cur_pos))
        pos[i] = cur_pos

    return pd.Series(pos, index=idx, name="pos")
