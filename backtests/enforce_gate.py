import sys, argparse, re

p = argparse.ArgumentParser()
p.add_argument('file', type=str)
p.add_argument('--min_sharpe', type=float, default=1.2)
p.add_argument('--maxdd', type=float, default=0.25)
args = p.parse_args()

text = open(args.file, 'r', encoding='utf-8').read()
def g(key, default):
    m = re.search(rf"{key}=([\d\.]+)", text)
    return float(m.group(1)) if m else default

sharpe = g("Sharpe", 0)
maxdd  = g("MaxDD", 1.0)

print(f"[gate] Sharpe={sharpe} (min {args.min_sharpe}), MaxDD={maxdd} (max {args.maxdd})")
if sharpe < args.min_sharpe or maxdd > args.maxdd:
    print("::error::Gate failed")
    sys.exit(1)
print("::notice::Gate passed")
