# config.py — central settings
from __future__ import annotations
import os
from pathlib import Path
from zoneinfo import ZoneInfo

# === Time zone (Bangkok) ===
LOCAL_TZ = ZoneInfo(os.getenv("LBOT_TZ", "Asia/Bangkok"))

# === Data path ===
LBOT_DATA_DIR = Path(os.getenv(
    "LBOT_DATA_DIR",
    r"C:\Users\WP_Bi\Trading Bot\DRL-bot\data"
)).resolve()

DATA_FILE = LBOT_DATA_DIR / "XAUUSD_15m_clean.csv"  # ใช้ชุด 'clean' ตามที่ระบุ

# === Trading / Risk caps ===
TAKER_FEE_BPS_PER_SIDE = float(os.getenv("LBOT_FEE_BPS", "0.5"))  # 0.5 bps ต่อขา
MIN_TRADES_PER_DAY = 3
RISK_PER_TRADE_PCT = 0.5
RISK_PER_DAY_PCT   = 2.0
MAX_POS_TOTAL      = 3
MAX_POS_PER_STRAT  = 3

# === Annualization for Sharpe (M15 bars) ===
M15_PER_DAY = 96
ANN_FACTOR  = 252 * M15_PER_DAY
