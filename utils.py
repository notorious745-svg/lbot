# utils.py
import numpy as np
import pandas as pd
from datetime import datetime
import os

def load_data(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def sharpe_ratio(returns, risk_free=0.0):
    excess_ret = np.array(returns) - risk_free
    if excess_ret.std() == 0:
        return 0
    return np.sqrt(252) * excess_ret.mean() / excess_ret.std()

def max_drawdown(equity_curve):
    arr = np.array(equity_curve)
    peak = arr[0]
    max_dd = 0
    for val in arr:
        peak = max(peak, val)
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)
    return max_dd

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
