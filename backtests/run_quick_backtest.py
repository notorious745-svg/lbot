from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --- optional deps inside repo ---
try:
    # ของโปรเจกต์เรา (ถ้ามี)
    from core.data_loader import load_price_csv  # type: ignore
except Exception:
    load_price_csv = None  # fallback ด้านล่างจะจัดการให้

from core.entries import combined_signal


def _load_prices(symbol: str, minutes: int) -> pd.DataFrame:
    """
    พยายามโหลดด้วย core.data_loader ก่อน
    ถ้าไม่มี ให้หาไฟล์ CSV มาตรฐานใน ./data/ (close/high/low/time)
    """
    if load_price_csv is not None:
        try:
            # รองรับ signature ที่ต่างกัน
            try:
                df = load_price_csv(symbol=symbol, minutes=minutes)  # type: ignore
            except TypeError:
                df = load_price_csv(symbol, minutes)  # type: ignore
            return df
        except Exception as e:
            print(f"[warn] load_price_csv failed: {e!r}", flush=True)

    data_dir = Path("data")
    # ชื่อไฟล์ที่มักใช้กัน
    candidates: List[Path] = [
        data_dir / f"{symbol}_15m_clean.csv",
        data_dir / f"{symbol}_15m.csv",
        data_dir / f"{symbol}.csv",
    ]
    for p in candidates:
        if p.exists():
            print(f"[info] loading {p}", flush=True)
            df = pd.read_csv(p)
            # ปรับชื่อคอลัมน์ให้มาตรฐาน
            cols = {c.lower(): c for c in df.columns}
            rename = {}
            for need in ("time", "timestamp", "datetime"):
                if need in cols:
                    rename[cols[need]] = "time"
                    break
            for need in ("close",):
                if need in cols:
                    rename[cols[need]] = "close"
            for need in ("high",):
                if need in cols:
                    rename[cols[need]] = "high"
            for need in ("low",):
                if need in cols:
                    rename[cols[need]] = "low"
            if rename:
                df = df.rename(columns=rename)
            # แปลงเวลา
            if "time" in df.columns:
                try:
                    df["time"] = pd.to_datetime(df["time"])
                except Exception:
                    pass
            if minutes and minutes > 0 and len(df) > minutes:
                df = df.iloc[-minutes:].copy()
            return df

    raise FileNotFoundError(
        "ไม่พบข้อมูลราคา: ใช้ core.data_loader ไม่ได้ และหา CSV ใน ./data/ ไม่เจอ"
    )


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_path = Path("backtests") / "out.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("• loading price...", flush=True)
    df = _load_prices(args.symbol, args.minutes)

    # ตั้งค่าคอลัมน์ close/high/low ให้พร้อมใช้
    if "close" not in df.columns:
        raise ValueError("input DataFrame ต้องมีคอลัมน์ 'close'")
    for need in ("high", "low"):
        if need not in df.columns:
            # ถ้าไม่มี ให้ใช้ close เป็นตัวแทน (จะทำให้กลยุทธ์พวก turtle ยังรันได้)
            df[need] = df["close"]

    nrow = len(df)
    print(f"• rows={nrow:,} minutes={args.minutes}", flush=True)

    # --- สร้างสัญญาณเทรด ---
    print("• building trading signal ...", flush=True)
    sig = combined_signal(
        df=df,
        strats=args.strats,
        atr_mult=args.atr_mult,
        vote=args.vote,
        cooldown=args.cooldown,
        session=args.session,
        max_layers=args.max_layers,
        pyr_step_atr=args.pyr_step_atr,
    ).astype(float).fillna(0.0)

    # --- คำนวณผลกำไรแบบง่าย (pnl ต่อบาร์) ---
    # ใช้ return แบบเปลี่ยนแปลงสัมพัทธ์ของราคาปิด
    ret = df["close"].pct_change().fillna(0.0)
    # ถือสถานะตาม signal ของบาร์ก่อนหน้า (หลีกเลี่ยง look-ahead)
    pos = sig.shift(1).fillna(0.0)
    pnl = (pos * ret).fillna(0.0)

    # equity curve (ตั้งต้นที่ 1.0)
    equity = (1.0 + pnl).cumprod()

    # นับจำนวนครั้งสลับสัญญาณเป็นจำนวน trade คร่าว ๆ
    trades = (pos.diff().abs() > 0).sum()

    # เตรียมผลลัพธ์ส่งออก
    out = pd.DataFrame(
        {
            "time": df["time"] if "time" in df.columns else np.arange(len(df)),
            "close": df["close"].values,
            "signal": sig.values,
            "ret": ret.values,
            "pnl": pnl.values,
            "equity": equity.values,
        }
    )

    print("• writing backtests/out.txt ...", flush=True)
    out.to_csv(out_path, index=False)

    dt = time.time() - t0
    print(f"• done. trades={int(trades)}  elapsed={dt:.1f}s", flush=True)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, default=60000)
    p.add_argument("--symbol", type=str, default="XAUUSD")
    p.add_argument("--session", type=str, default="all")  # เช่น ln_ny, all
    p.add_argument("--strats", type=str, default="ema,turtle20,turtle55")
    p.add_argument("--atr_mult", type=float, default=3.0)
    p.add_argument("--max_layers", type=int, default=0)
    p.add_argument("--vote", type=int, default=1)
    p.add_argument("--cooldown", type=int, default=0)
    p.add_argument("--pyr_step_atr", type=float, default=1.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args)
