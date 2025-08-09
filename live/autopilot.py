import os, sys, numpy as np, pandas as pd
from datetime import datetime, timezone
from stable_baselines3 import PPO
from pathlib import Path

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import features (project-first)
try:
    from features import build_state_features as _build
except Exception:
    from features import add_indicators, normalize_features
    def _build(df: pd.DataFrame) -> pd.DataFrame:
        df = add_indicators(df)
        df = normalize_features(df)
        num = df.select_dtypes(include=[np.number]).columns
        return df[num]

from live.bridge.ctrader_bridge import (
    connect, subscribe, on_bar_close, positions, place, modify_sl, close, price, run_sim_from_csv
)

SYMBOL    = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME = os.getenv("TIMEFRAME", "M15")
DEMO     = os.getenv("DEMO_MODE", "true").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "models/drl_model.zip")

ATR_PERIOD     = int(os.getenv("ATR_PERIOD", "14"))
ATR_SL_MULT    = float(os.getenv("ATR_SL_MULT", "2.0"))
ATR_TRAIL_MULT = float(os.getenv("ATR_TRAIL_MULT", "1.0"))
BE_AFTER_ATR   = float(os.getenv("BE_AFTER_ATR", "0.5"))
RISK_PCT_TRADE = float(os.getenv("RISK_PCT_TRADE", "0.005"))
LOT_SIZE       = float(os.getenv("LOT_SIZE", "1.0"))

def _atr(hist: pd.DataFrame, period: int=14) -> float:
    h,l,c = hist["high"], hist["low"], hist["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def main():
    print("[AP] Loading model:", MODEL_PATH)
    try:
        model = PPO.load(MODEL_PATH)
    except Exception:
        print("[AP] Model not found or load failed. Using RANDOM policy for demo.")
        class _Rand:
            def predict(self, obs, deterministic=True):
                return np.random.choice([0,1,2]), None
        model = _Rand()

    connect(); subscribe(SYMBOL, TIMEFRAME)

    def handle_close(bar: dict):
        hist = bar["history"]
        if len(hist) < 120: return

        feats = _build(hist.copy())
        obs = feats.iloc[-1].to_numpy(dtype=np.float32).ravel()

        act, _ = model.predict(obs, deterministic=True)
        act = int(np.asarray(act).flatten()[0])  # 0=hold,1=long,2=short

        px = float(bar["close"])
        a  = _atr(hist, ATR_PERIOD)
        if not np.isfinite(a) or a <= 0: return

        # risk sizing
        sl_dist = ATR_SL_MULT * a
        risk_cash = 10_000 * RISK_PCT_TRADE  # TODO: query live equity from bridge
        volume = max(risk_cash / max(sl_dist*LOT_SIZE,1e-9), 0.01)

        # cap + round
        volume = float(np.clip(volume, 0.01, 1.00))
        volume = round(volume, 2)

        side = {1:"BUY", 2:"SELL"}.get(act)
        if side is None:
            # TODO: trailing/breakeven mgmt on existing positions
            return

        sl = px - sl_dist if side=="BUY" else px + sl_dist
        place(side, volume, sl=sl, tp=None, label="AP")
        print(f"[DEMO {DEMO}] {bar['time']:%Y-%m-%d %H:%M} {side} vol={volume:.2f} sl={sl:.2f} @ {px:.2f}")

        # log to CSV
        import os, csv
        os.makedirs("logs", exist_ok=True)
        with open("logs/signals.csv", "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["time","side","volume","price","sl","atr"])
            w.writerow([bar["time"], side, volume, px, sl, a])

    on_bar_close(handle_close)

    # SIM from CSV if no realtime bridge
    csv_path = os.getenv("CSV_PATH", "data/XAUUSD_15m_sample.csv")
    if os.path.exists(csv_path):
        print("[AP] SIM from CSV:", csv_path)
        run_sim_from_csv(csv_path, sleep_sec=0.05)
    else:
        print("[AP] Waiting realtime bars...")

if __name__ == "__main__":
    main()
