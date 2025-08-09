# reward_fn.py
import numpy as np
from utils import sharpe_ratio, max_drawdown
from config import TARGET_SHARPE, REWARD_SCALING

def reward_from_trade(pnl_list, equity_curve):
    if len(pnl_list) < 2:
        return 0.0

    sharpe = sharpe_ratio(pnl_list)
    mdd = max_drawdown(equity_curve)

    # Reward shaping: encourage high Sharpe, low DD
    reward = (sharpe / TARGET_SHARPE) - mdd
    return reward * REWARD_SCALING

def step_reward(pnl, position_size):
    # Simple risk-adjusted reward
    return np.sign(pnl) * np.sqrt(abs(pnl) / max(position_size, 1e-6))
