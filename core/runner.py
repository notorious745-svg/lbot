from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd
from config import LOCAL_TZ, LBOT_DATA_DIR, DATA_FILE
from core.data_loader import load_price_csv
from core.entries import combined_signal
from core.position_manager import build_orders

M15 = timedelta(minutes=15)
def floor_to_m15(dt: datetime) -> datetime:
    q = (dt.minute // 15) * 15
    return dt.replace(minute=q, second=0, microsecond=0, tzinfo=LOCAL_TZ)

if __name__ == "__main__":
    print(f"[i] TZ={LOCAL_TZ}  DATA_DIR={LBOT_DATA_DIR}  FILE={DATA_FILE}")
    df = load_price_csv()
    sig = combined_signal(df)
    plan = build_orders(sig)

    last = df.iloc[-1]
    print(f"[i] rows={len(df)} last_bkk={last['time']} close={last['close']}")
    print(f"[i] last M15 bucket (BKK) = {floor_to_m15(last['time'].to_pydatetime())}")

    tail = pd.concat([
        df[["time","close"]].tail(10).reset_index(drop=True),
        sig.tail(10).reset_index(drop=True),
        plan.tail(10).reset_index(drop=True)
    ], axis=1)
    print(tail.to_string(index=False))
    print("[ok] runner finished (demo-safe if CSV missing)")
