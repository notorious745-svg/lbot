import argparse, json, random, time, sys

parser = argparse.ArgumentParser()
parser.add_argument('--minutes', type=int, default=5000)
parser.add_argument('--symbol', type=str, default='XAUUSD')
args = parser.parse_args()

# Dummy quick backtest stub — replace with your engine.
random.seed(42)
metrics = {
  "symbol": args.symbol,
  "minutes": args.minutes,
  "net_pnl": round(random.uniform(500, 2000), 2),
  "sharpe": round(random.uniform(1.3, 2.5), 2),
  "max_dd": round(random.uniform(0.10, 0.22), 2),
  "trades_per_day": round(random.uniform(6, 14), 1)
}
print(json.dumps(metrics, ensure_ascii=False))
