from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# โมดูลภายในโปรเจกต์ (เขียนให้ยืดหยุ่นต่อ signature)
from core import entries as E
from core.spike_filter import spike_flag

# data_loader / position_manager ในรีโปอาจมี signature ต่างกัน
try:
    from core.data_loader import load_price_csv  # type: ignore
except Exception:  # pragma: no cover
    load_price_csv = None  # จะใช้ fallback อ่าน CSV ตรง ๆ

try:
    from core.position_manager import generate_position_series  # type: ignore
except Exception:  # pragma: no cover
    generate_position_series = None


def _filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """ส่งเฉพาะ kwargs ที่ฟังก์ชันนั้นรองรับ (กัน TypeError)"""
    if func is None:
        return {}
    try:
        params = inspect.signature(func).parameters
        return {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        return kwargs


def _load_data(symbol: str, minutes: int | None) -> pd.DataFrame:
    """
    พยายามโหลดข้อมูลผ่าน core.data_loader.load_price_csv ก่อน
    ถ้าไม่ได้ ให้หาไฟล์ที่คุ้น ๆ (เช่น data/XAUUSD_15m_clean.csv) เป็น fallback
    """
    if load_price_csv is not None:
        # พยายามเดา signature ที่พบได้บ่อย
        for sig in (
            {"symbol": symbol, "minutes": minutes},
            {"symbol": symbol},
            {"minutes": minutes},
            {},
        ):
            try:
                df = load_price_csv(**_filter_kwargs(load_price_csv, sig))  # type: ignore
                if isinstance(df, pd.DataFrame):
                    return df.copy()
            except Exception:
                pass

    # Fallback: เดาไฟล์ CSV ชื่อที่น่าจะใช้งาน
    candidates = [
        Path("data") / f"{symbol}_15m_clean.csv",
        Path("data") / f"{symbol}_15m.csv",
        Path("data") / f"{symbol}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(
            "ไม่พบไฟล์ราคา และเรียกใช้ load_price_csv ไม่สำเร็จ "
            "กรุณาใส่ไฟล์ราคาลงในโฟลเดอร์ data/ เช่น XAUUSD_15m_clean.csv"
        )

    # ทำให้คอลัมน์มาตรฐาน
    cols = {c.lower(): c for c in df.columns}
    for need in ("time", "open", "high", "low", "close"):
        if need not in {c.lower() for c in df.columns}:
            raise ValueError(f"ต้องมีคอลัมน์ {need} ในไฟล์ราคา")
    # ปรับชื่อเป็น lower
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
        except Exception:
            pass
    return df


def run(args: argparse.Namespace) -> pd.DataFrame:
    df = _load_data(args.symbol, args.minutes)

    # ทำสัญญาณรวม (ยืดหยุ่นต่อ interface)
    strat_list = [s.strip() for s in str(args.strats).split(",") if s.strip()]
    sig_kwargs = dict(
        strats=strat_list,
        atr_n=args.atr_n,
        atr_mult=args.atr_mult,
        vote=args.vote,
        cooldown=args.cooldown,
        session=args.session,
    )
    sig = E.combined_signal(
        df,
        **_filter_kwargs(E.combined_signal, sig_kwargs),
    ).astype(float)

    # ปิดสัญญาณที่เป็น spike (ถ้ามี)
    try:
        mask_spike = spike_flag(df, n=args.atr_n, k=args.atr_mult)
        sig[mask_spike > 0] = 0.0
    except Exception:
        pass

    # แปลงสัญญาณเป็นตำแหน่ง (ใช้ position_manager ถ้ามี)
    if generate_position_series is not None:
        pm_kwargs = dict(
            d=df,
            signal=sig,
            atr_step=args.pyr_step_atr,
            max_layers=args.max_layers,
            cooldown=args.cooldown,
        )
        pos = generate_position_series(**_filter_kwargs(generate_position_series, pm_kwargs))
    else:
        # fallback: ใช้สัญญาณตรง ๆ
        pos = sig.copy()

    pos = pd.Series(pos, index=df.index).fillna(0.0)
    ret = df["close"].pct_change().fillna(0.0) * pos.shift(1).fillna(0.0)
    equity = (1.0 + ret).cumprod()

    # เขียนผลลัพธ์แบบเรียบง่าย (CSV)
    out = pd.DataFrame(
        {
            "time": df.index,
            "close": df["close"].values,
            "signal": sig.values,
            "pos": pos.values,
            "equity": equity.values,
        }
    )
    out_path = Path("backtests") / "out.txt"
    out.to_csv(out_path, index=False)

    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, default=5000)
    p.add_argument("--symbol", type=str, default="XAUUSD")
    p.add_argument("--session", type=str, default="all")
    p.add_argument("--strats", type=str, default="ema,turtle20,turtle55")
    p.add_argument("--atr_n", type=int, default=14)
    p.add_argument("--atr_mult", type=float, default=3.0)
    p.add_argument("--max_layers", type=int, default=0)
    p.add_argument("--pyr_step_atr", type=float, default=1.0)
    p.add_argument("--vote", type=int, default=1)
    p.add_argument("--cooldown", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args)
