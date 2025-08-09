# config.py

# === Basic Trading Config ===
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"

INITIAL_BALANCE = 10000
LOT_SIZE = 0.01
MAX_STEPS = 500_000  # safety limit

# === DRL Agent Config ===
AGENT_TYPE = "PPO"  # or "A2C", "DQN"
POLICY_TYPE = "MlpPolicy"
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_STEPS = 2048
BATCH_SIZE = 256

# === Sharpe Optimization ===
TARGET_SHARPE = 3.0
REWARD_SCALING = 1.0  # base multiplier for reward

# === Market Regime Detection ===
REGIME_LOOKBACK = 55
REGIME_METHOD = "ema_spread"  # "ema_spread", "volatility", etc.
EMA_PERIODS = (10, 25, 50)
EMA_SPREAD_THRESHOLD = 0.3

# === Feature Engineering ===
ATR_PERIOD = 14
RSI_PERIOD = 14
TURTLE_ENTRY_1 = 20
TURTLE_EXIT_1 = 10
TURTLE_ENTRY_2 = 55
TURTLE_EXIT_2 = 20

# === News Filter ===
USE_NEWS_FILTER = True
NEWS_IMPACT_THRESHOLD = 2  # 1=low, 2=medium, 3=high

# === File Paths ===
DATA_PATH = "data/XAUUSD_15m.csv"
MODEL_PATH = "models/drl_model.zip"
LOG_DIR = "logs/"
