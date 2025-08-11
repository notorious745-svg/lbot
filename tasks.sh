#!/usr/bin/env bash
set -e
cmd="$1"; shift || true
case "$cmd" in
  lint)    python -m ruff check src ;;
  train)   python train/train.py --epochs 3 "$@" ;;
  bt)      python backtests/run_quick_backtest.py --minutes 5000 --symbol XAUUSD "$@" ;;
  metrics) python backtests/print_metrics.py backtests/out.txt ;;
  gate)    python backtests/enforce_gate.py --maxdd 0.25 --min_sharpe 1.2 metrics.txt ;;
  push)    git add -A && git commit -m "${1:-chore}" && git push ;;
  *) echo "usage: ./tasks.sh [lint|train|bt|metrics|gate|push]"; exit 1 ;;
esac
