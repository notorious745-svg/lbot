# scripts/make_demo_data.py
import os, numpy as np, pandas as pd
os.makedirs("data", exist_ok=True)

N = 5000
t = pd.date_range("2024-01-01", periods=N, freq="15min")
close = 2000 + np.cumsum(np.random.normal(0, 1.2, size=N))
open_  = np.r_[close[0], close[:-1]]
high   = np.maximum(open_, close) + np.random.rand(N) * 0.8
low    = np.minimum(open_, close) - np.random.rand(N) * 0.8
vol    = np.random.randint(100, 1000, size=N)

df = pd.DataFrame({
    "time": t,
    "open": open_,
    "high": high,
    "low":  low,
    "close": close,
    "volume": vol,
})
df.to_csv("data/xauusd_15m_demo.csv", index=False)
print("Wrote data/xauusd_15m_demo.csv", len(df), "rows")
