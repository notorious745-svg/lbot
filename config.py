from __future__ import annotations
import os
from dataclasses import dataclass

# --------- Symbol / timeframe ----------
SYMBOL = "XAUUSD"
TIMEFRAME_MINUTES = 15  # เราทำงานบน M15 เท่านั้น

# --------- Paths ----------
# ใช้ตัวแปรแวดล้อมถ้ามี ไม่งั้น fallback เป็นโฟลเดอร์ data ใน repo
DATA_DIR = os.environ.get("LBOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DATA_FILE = os.path.join(DATA_DIR, f"{SYMBOL}_15m_clean.csv")  # ชื่อไฟล์มาตรฐาน

# --------- Risk guard (manual mode ดีฟอลต์) ----------
RISK_PER_TRADE = 0.005        # 0.5%/trade
DAILY_RISK_CAP = 0.02         # 2%/day
MAX_OPEN_POSITIONS_TOTAL = 3  # รวมทุกกลยุทธ์
MAX_OPEN_POSITIONS_PER_STRAT = 3

# --------- Filters ----------
SPIKE_ATR_MULT = 3.0          # ถ้า range > 3*ATR14 → skip สัญญาณ
ATR_PERIOD = 14

# --------- Entries / Exits ----------
EMA_FAST = 10
EMA_SLOW = 20
EMA_TREND = 50

# trailing/exit hook (สำหรับต่อ ML ภายหลัง)
ATR_TRAIL_MULT = 3.0
BREAKEVEN_AFTER_R_MULT = 1.0  # ย้าย SL เป็น BE เมื่อวิ่งได้ 1R

# --------- Misc ----------
SEED = 2025
BAR_CLOSE_ONLY = True  # เข้าสัญญาณเฉพาะตอนแท่งปิด (M15)
