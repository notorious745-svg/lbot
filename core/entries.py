from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .spike_filter import spike_flag

# พยายามใช้ indicators ของโปรเจกต์ ถ้าไม่มีให้ทำแบบง่ายในไฟล์นี้
try:
    from .indicators import ema  # type: ignore
except Exception:
    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()


def _turtle_breakout(df: pd.DataFrame, n: int) -> pd.Series:
    """+1 เมื่อราคาปิดทะลุ High(n) / -1 เมื่อราคาปิดหลุด Low(n) / 0 อื่น ๆ"""
    hi = df["high"].rolling(n, min_periods=1).max()
    lo = df["low"].rolling(n, min_periods=1).min()
    c = df["close"]
    up = (c > hi.shift(1)).astype(int)
    dn = (c < lo.shift(1)).astype(int) * -1
    sig = up + dn
    sig = sig.replace({2: 1, -2: -1})
    return sig.reindex(df.index).fillna(0)


def _ema_cross(df: pd.DataFrame, fast: int = 20, slow: int = 55) -> pd.Series:
    f = ema(df["close"], fast)
    s = ema(df["close"], slow)
    sig = np.where(f > s, 1, -1)
    return pd.Series(sig, index=df.index, dtype=float)


def _majority_vote(signals: Iterable[pd.Series], vote: int) -> pd.Series:
    # รวมคะแนน แล้ว threshold ตาม vote
    sig_mat = np.column_stack([s.fillna(0).values for s in signals])
    score = sig_mat.sum(axis=1)
    out = np.where(score >= vote, 1, np.where(score <= -vote, -1, 0))
    return pd.Series(out, index=signals.__iter__().__next__().index, dtype=float)


def combined_signal(
    df: pd.DataFrame,
    strats: str = "ema,turtle20,turtle55",
    atr_mult: float = 3.0,   # ยังไม่ได้ใช้ในสูตรง่าย ๆ นี้ แต่เก็บพารามิเตอร์ไว้ก่อน
    vote: int = 1,
    cooldown: int = 0,
    session: str = "all",
    max_layers: int = 0,
    pyr_step_atr: float = 1.0,
) -> pd.Series:
    """
    รวมสัญญาณแบบง่าย:
      - ema          : EMA(20) cross EMA(55)
      - turtle20     : breakout 20
      - turtle55     : breakout 55
    จากนั้นทำ majority vote ด้วย 'vote'
    และ mask spikes ด้วย spike_flag
    """
    s_names = [x.strip().lower() for x in strats.split(",") if x.strip()]
    sigs: list[pd.Series] = []

    if "ema" in s_names:
        sigs.append(_ema_cross(df, 20, 55))
    if "turtle20" in s_names:
        sigs.append(_turtle_breakout(df, 20))
    if "turtle55" in s_names:
        sigs.append(_turtle_breakout(df, 55))

    if not sigs:
        # ถ้าไม่เลือกอะไรเลย ให้ถือศูนย์ทั้งเส้น
        base = pd.Series(0.0, index=df.index)
        return base

    sig = _majority_vote(sigs, max(1, vote))

    # cooldown: บังคับให้คง signal เดิมไว้ n บาร์หลังเปลี่ยนสถานะ
    if cooldown and cooldown > 0:
        last = 0.0
        hold = 0
        buf = []
        for v in sig.values:
            if v != 0 and np.sign(v) != np.sign(last):
                last = v
                hold = cooldown
            elif hold > 0:
                v = last
                hold -= 1
            buf.append(v)
        sig = pd.Series(buf, index=sig.index, dtype=float)

    # spike mask: ถ้า spike ให้เป็น 0
    mask = spike_flag(df, n=14, k=3.0)
    sig = sig.where(mask == 0, 0.0)

    return sig.astype(float).fillna(0.0)
