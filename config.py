# config.py
import os
from dataclasses import dataclass

# ชี้ไฟล์ข้อมูล: ใช้ ENV ก่อน ถ้าไม่ตั้งจะ fallback ค่า default ด้านล่าง
DATA_FILE = os.getenv(
    "LBOT_DATA_FILE",
    # เปลี่ยนเป็นไฟล์ที่คุณมี หรือปล่อยไว้ก่อนก็ได้
    "data/xauusd_15m_clean.csv",
)

# ===== Risk / Trade params (ปรับตามสบาย) =====
BAR_CLOSE_ONLY = True          # ตัดสินใจตอนแท่งปิด
RISK_PER_TRADE = 0.005         # 0.5% ของ equity ต่อไม้
DAILY_RISK_CAP = 0.02          # max risk ต่อวัน
MAX_PYRAMID = 3                # จำนวนไม้เพิ่มสูงสุดเมื่อวิ่งทางกำไร
ADD_ON_GAIN_R_MULT = 0.8       # เพิ่มไม้เมื่อกำไร >= 0.8R
BREAKEVEN_AFTER_R_MULT = 1.0   # ย้าย SL ไป BE เมื่อกำไรถึง 1R
ATR_PERIOD = 14
ATR_TRAIL_MULT = 2.0

# ===== Session / Filter (ใช้หรือไม่ใช้ก็ได้) =====
USE_SPIKE_FILTER = True
SPIKE_MULT = 2.2                # body > ATR*k -> ถือว่า spike

# ===== Backtest =====
START_BALANCE = 10_000.0
SLIPPAGE = 0.0
COMMISSION_PER_TRADE = 0.0

@dataclass
class Symbols:
    name: str = "XAUUSD"
    pip: float = 0.01          # ใช้คำนวณ R/ATR sizing
SYMBOL = Symbols()
