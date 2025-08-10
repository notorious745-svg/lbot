# live/autopilot.py — run PPO live/sim with obs-dim fix
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

# -------------------------------------------------
# ใส่โฟลเดอร์โปรเจ็กต์ลง sys.path (ให้ import โมดูลภายในได้)
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# ดึงตัวสร้างฟีเจอร์ (ถ้าไม่มี build_state_features ให้ fallback)
# -------------------------------------------------
try:
    from features import build_state_features as _build
except Exception:
    from features import add_indicators, normalize_features

    def _build(df: pd.DataFrame) -> pd.DataFrame:
        df = add_indicators(df)
        df = normalize_features(df)
        num = df.select_dtypes(include=[np.number]).columns
        return df[num]

# -------------------------------------------------
# สะพานเชื่อม cTrader (หรือ mock ที่แถมมา)
# -------------------------------------------------
from live.bridge.ctrader_bridge import (
    connect,
    subscribe,
    on_bar_close,
    positions,
    place,
    modify_sl,
    close,
    price,
    run_sim_from_csv,
)

# -------------------------------------------------
# ค่าคอนฟิกผ่าน ENV
# -------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME = os.getenv("TIMEFRAME", "M15")
DEMO = os.getenv("DEMO_MODE", "true").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "models/drl_model.zip")

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "2.0"))
ATR_TRAIL_MULT = float(os.getenv("ATR_TRAIL_MULT", "1.0"))
BE_AFTER_ATR = float(os.getenv("BE_AFTER_ATR", "0.5"))
RISK_PCT_TRADE = float(os.getenv("RISK_PCT_TRADE", "0.005"))
LOT_SIZE = float(os.getenv("LOT_SIZE", "1.0"))

# -------------------------------------------------
# ตัวช่วยคำนวณ ATR ง่าย ๆ
# -------------------------------------------------
def _atr(hist: pd.DataFrame, period: int = 14) -> float:
    h, l, c = hist["high"], hist["low"], hist["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

# -------------------------------------------------
# FIX: บังคับขนาด observation ให้เท่ากับที่โมเดลต้องการ (เช่น 16)
# -------------------------------------------------
def _model_obs_dim(m) -> int | None:
    try:
        return int(m.observation_space.shape[0])
    except Exception:
        try:
            return int(m.policy.observation_space.shape[0])
        except Exception:
            return None

def _fix_obs_dim(m, obs: np.ndarray) -> np.ndarray:
    n_model = _model_obs_dim(m)
    # ถ้ารู้มิติที่โมเดลต้องการ ให้แพด/ตัดให้พอดี
    if n_model is not None:
        obs = np.asarray(obs, dtype=np.float32).ravel()
        if obs.shape[0] == n_model:
            return obs
        fixed = np.zeros(n_model, dtype=np.float32)
        k = min(n_model, obs.shape[0])
        fixed[:k] = obs[:k]
        return fixed
    # ถ้าดึงไม่ได้ก็ส่งแบบเดิม
    return np.asarray(obs, dtype=np.float32).ravel()

# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    print("[AP] Loading model:", MODEL_PATH)
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print("[AP] Model not found or load failed. Using RANDOM policy for demo.", e)

        class _Rand:
            def predict(self, obs, deterministic=True):
                return np.random.choice([0, 1, 2]), None

        model = _Rand()

    connect()
    subscribe(SYMBOL, TIMEFRAME)

    def handle_close(bar: dict):
        hist = bar["history"]
        if len(hist) < 120:
            return

        # สร้างฟีเจอร์ → ดึงแถวล่าสุด → ทำให้มิติ obs ตรงกับโมเดล
        feats = _build(hist.copy())
        obs = feats.iloc[-1].to_numpy(dtype=np.float32).ravel()
        obs = _fix_obs_dim(model, obs)

        # ทำนายแอคชัน 0=Hold 1=BUY 2=SELL
        act, _ = model.predict(obs, deterministic=True)
        act = int(np.asarray(act).flatten()[0])

        px = float(bar["close"])
        a = _atr(hist, ATR_PERIOD)
        if not np.isfinite(a) or a <= 0:
            return

        # position sizing (เดโม่: ใช้ equity สมมติ 10k)
        sl_dist = ATR_SL_MULT * a
        risk_cash = 10_000 * RISK_PCT_TRADE
        volume = max(risk_cash / max(sl_dist * LOT_SIZE, 1e-9), 0.01)
        volume = float(np.clip(volume, 0.01, 1.00))
        volume = round(volume, 2)

        side = {1: "BUY", 2: "SELL"}.get(act)
        if side is None:
            # TODO: trailing stop / breakeven สำหรับโพสิชันที่ถืออยู่
            return

        sl = px - sl_dist if side == "BUY" else px + sl_dist
        place(side, volume, sl=sl, tp=None, label="AP")
        print(f"[DEMO {DEMO}] {bar['time']:%Y-%m-%d %H:%M} {side} vol={volume:.2f} sl={sl:.2f} @ {px:.2f}")

        # บันทึกสัญญาณลง CSV
        import os, csv

        os.makedirs("logs", exist_ok=True)
        with open("logs/signals.csv", "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["time", "side", "volume", "price", "sl", "atr"])
            w.writerow([bar["time"], side, volume, px, sl, a])

    on_bar_close(handle_close)

    # ถ้ามี CSV ให้รัน SIM ทันที (ไม่มี feed จริงก็ยังเทสได้)
    csv_path = os.getenv("CSV_PATH", "data/XAUUSD_15m_sample.csv")
    if os.path.exists(csv_path):
        print("[AP] SIM from CSV:", csv_path)
        run_sim_from_csv(csv_path, sleep_sec=0.05)
    else:
        print("[AP] Waiting realtime bars...")

if __name__ == "__main__":
    main()
