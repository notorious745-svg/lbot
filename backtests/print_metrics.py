import sys, json
text = open(sys.argv[1], 'r', encoding='utf-8').read()
try:
    j = json.loads(text.strip().splitlines()[-1])
except Exception:
    # fallback: metrics are unknown; force fail later
    j = {"net_pnl": 0, "sharpe": 0, "max_dd": 1.0, "trades_per_day": 0}
out = f"NetPnL={j['net_pnl']}\nSharpe={j['sharpe']}\nMaxDD={j['max_dd']}\nTPD={j['trades_per_day']}\n"
open('metrics.txt','w',encoding='utf-8').write(out)
print(out, end='')
