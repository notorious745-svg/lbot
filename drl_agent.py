# drl_agent.py — Train PPO + print full stats (Trades, PF, Expectancy, Sharpe 15m & Daily)
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import TradingEnv
from config import TOTAL_TIMESTEPS, MODEL_PATH, LOG_DIR


def _to_int_action(a):
    arr = np.asarray(a)
    return int(arr.flatten()[0]) if arr.size else 0


def _equity_metrics(time_index, equity_curve):
    eq = np.asarray(equity_curve, dtype=float)

    # Sharpe จาก 15m bar
    r_bar = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    sharpe_15m = np.sqrt(24 * 4 * 252) * (r_bar.mean() / (r_bar.std() + 1e-12))

    # Daily Sharpe (น่าเชื่อถือกว่า)
    import pandas as pd
    ts = pd.Series(eq, index=pd.to_datetime(time_index[: len(eq)]))
    daily_eq = ts.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe_daily = np.sqrt(252) * (daily_ret.mean() / (daily_ret.std() + 1e-12))

    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / np.maximum(peak, 1e-12)).min()
    net = eq[-1] - eq[0]
    return float(sharpe_15m), float(sharpe_daily), float(mdd), float(net)


def train():
    env = DummyVecEnv([lambda: TradingEnv()])
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=4096,
        ent_coef=0.1,
    )
    print(f"[PPO] Start learning for {TOTAL_TIMESTEPS:,} timesteps")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"Model saved => {MODEL_PATH}")
    return model


def evaluate(model):
    print("\n=== EVALUATION (deterministic) ===")
    env = TradingEnv()

    # reset: รองรับทั้ง Gym / Gymnasium
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out

    equity = [env.equity if hasattr(env, "equity") else env.balance]
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        step_out = env.step(_to_int_action(act))
        if len(step_out) == 5:   # Gymnasium: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:                    # Gym: (obs, reward, done, info)
            obs, reward, done, info = step_out
        equity.append(env.equity if hasattr(env, "equity") else env.balance)

    # ----- Trade-level stats -----
    trades = np.asarray(getattr(env, "trade_pnls", []), dtype=float)
    n = int(trades.size)
    if n > 0:
        wins = int((trades > 0).sum())
        losses = int((trades < 0).sum())
        win_rate = wins / n
        avg_win = trades[trades > 0].mean() if wins else 0.0
        avg_loss = (-trades[trades < 0]).mean() if losses else 0.0
        gross_win = trades[trades > 0].sum() if wins else 0.0
        gross_loss = -trades[trades < 0].sum() if losses else 0.0
        pf = (gross_win / max(gross_loss, 1e-12)) if losses else float("inf")
        expectancy = trades.mean()
        print(
            f"Trades: {n} | Win%: {win_rate:.2%} | PF: {pf:.2f} | "
            f"Expectancy: {expectancy:.2f} | AvgWin: {avg_win:.2f} | AvgLoss: {avg_loss:.2f}"
        )
    else:
        print("Trades: 0")

    # ----- Equity stats -----
    s15, sD, mdd, net = _equity_metrics(getattr(env, "_time", np.arange(len(equity))), equity)
    print(f"Sharpe (15m): {s15:.4f}")
    print(f"Sharpe (Daily): {sD:.4f}   <-- ใช้ค่านี้ตัดสินลงสนามจริง")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Net Profit  : {net:.2f}")


if __name__ == "__main__":
    mdl = train()
    evaluate(mdl)
