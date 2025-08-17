from __future__ import annotations

from typing import Iterable, List
import numpy as np
import pandas as pd

from core.spike_filter import spike_flag


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _ema_cross(df: pd.DataFrame, fast: int = 20, slow: int = 55) -> pd.Series:
    e1 = _ema(df["close"], fast)
    e2 = _ema(df["close"], slow)
    sig = pd.Series(0.0, index=df.index)
    sig[e1 > e2] = 1.0
    sig[e1 < e2] = -1.0
    return sig.ffill().fillna(0.0)


def _turtle_breakout(df: pd.DataFrame, n: int = 20) -> pd.Series:
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    sig = pd.Series(0.0, index=df.index)
    sig[df["close"] > hh.shift(1)] = 1.0
    sig[df["close"] < ll.shift(1)] = -1.0
    return sig.ffill().fillna(0.0)


def _majority_vote(sigs: List[pd.Series], vote: int) -> pd.Series:
    # ลงคะแนน -1/0/+1 แล้วรวม ถ้า |sum| >= vote ให้เอาทิศทาง sign(sum) มิฉะนั้น 0
    s = pd.concat(sigs, axis=1).fillna(0.0)
    sm = s.sum(axis=1)
    out = pd.Series(0.0, index=s.index)
    out[sm >= vote] = 1.0
    out[sm <= -vote] = -1.0
    return out


def _apply_cooldown(sig: pd.Series, cooldown: int) -> pd.Series:
    if cooldown <= 0:
        return sig
    sig = sig.copy()
    last = 0.0
    cd = 0
    for i, v in enumerate(sig.values):
        if v != 0 and np.sign(v) != np.sign(last):
            last = v
            cd = cooldown
        elif cd > 0:
            # ปิดการเปลี่ยนสถานะระหว่าง cooldown
            sig.iloc[i] = last
            cd -= 1
        else:
            last = v
    return sig


def combined_signal(
    d: pd.DataFrame,
    strats: Iterable[str] | None = None,
    atr_n: int = 14,
    atr_mult: float = 3.0,
    vote: int = 1,
    cooldown: int = 0,
    session: str = "all",
) -> pd.Series:
    """
    คืนค่า entry signal แบบ -1/0/+1
    - strats: ชื่อกลยุทธ์ เช่น ['ema','turtle20','turtle55']
    - vote: จำนวนเสียงที่ต้องถึง (majority)
    - cooldown: จำนวนแท่งที่จะคงสถานะหลังสลับฝั่ง
    """
    if strats is None:
        strats = ["ema", "turtle20", "turtle55"]

    sigs: List[pd.Series] = []

    for s in strats:
        s = s.strip().lower()
        if s == "ema":
            sigs.append(_ema_cross(d, 20, 55))
        elif s.startswith("turtle"):
            # ดึงตัวเลข n จากชื่อ เช่น turtle20, turtle55
            n = 20
            try:
                n = int("".join(ch for ch in s if ch.isdigit()) or "20")
            except Exception:
                pass
            sigs.append(_turtle_breakout(d, n))
        else:
            # ไม่รู้จักชื่อ -> ให้ศูนย์
            sigs.append(pd.Series(0.0, index=d.index))

    vote = max(1, int(vote))
    sig = _majority_vote(sigs, vote)

    # เซสชัน: เพื่อความง่าย เวอร์ชันนี้ยังไม่ตัดชั่วโมงเทรด (คงไว้ทั้งหมด)
    # ถ้าต้องการภายหลังค่อยเติม mask ตาม session

    # ตัด spike ออก (ป้องกันความผันผวนจัด)
    try:
        sp = spike_flag(d, n=atr_n, k=atr_mult)
        sig[sp > 0] = 0.0
    except Exception:
        pass

    sig = _apply_cooldown(sig, cooldown)
    return sig.astype(float)
