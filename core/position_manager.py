from __future__ import annotations
import pandas as pd
from config import MAX_POS_TOTAL, MAX_POS_PER_STRAT

# โครงแบบง่าย: แปลงสัญญาณเป็น position target ตามกฎ caps/pyramid เฉพาะกำไร
# (ปล่อย logic pyramid ละเอียด + magic grouping ให้ใส่เพิ่มเติมตอนต่อ broker)

STRATS = ["ema", "turtle20", "turtle55", "meanrev"]

def apply_caps(sig: pd.DataFrame) -> pd.DataFrame:
    """
    รับสัญญาณต่อบาร์ (−1/0/+1) ต่อกลยุทธ์ → คืนเป้าหมาย pos_total & per-strategy
    """
    pos = sig[STRATS].copy()
    # clamp ต่อกลยุทธ์
    pos = pos.clip(-1, 1)  # ต่อกลยุทธ์สูงสุด 1 หน่วย (ภายหลัง map เป็นจำนวนไม้อีกที)
    # รวมทุกกลยุทธ์ แล้ว clamp รวมระบบ
    pos["sum"] = pos.sum(axis=1)
    pos["sum"] = pos["sum"].clip(-MAX_POS_TOTAL, MAX_POS_TOTAL)
    return pos

def build_orders(sig: pd.DataFrame) -> pd.DataFrame:
    """
    แปลง target pos เป็น 'turning points' เพื่อเปิด/ปิด (ยังไม่ยิงคำสั่งจริง)
    """
    pos = apply_caps(sig)
    turns = pos[STRATS].diff().fillna(0)
    orders = (turns != 0).astype(int)
    orders.columns = [f"order_{c}" for c in orders.columns]
    return pd.concat([pos, orders], axis=1)
