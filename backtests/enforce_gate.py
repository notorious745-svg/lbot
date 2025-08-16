from __future__ import annotations
import argparse, sys, re

def read_metrics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    def pick(key, default):
        m = re.search(rf"{key}\s*=\s*([\-+]?\d+(\.\d+)?)", txt, flags=re.IGNORECASE)
        return float(m.group(1)) if m else default
    return {
        "sharpe": pick("sharpe", 0.0),
        "maxdd":  pick("maxdd", 1.0),
        "trades": int(pick("trades", 0.0)),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_file")
    ap.add_argument("--min_sharpe", type=float, required=True)
    ap.add_argument("--maxdd", type=float, required=True)
    args = ap.parse_args()

    m = read_metrics(args.metrics_file)
    ok = True
    if m["sharpe"] < args.min_sharpe:
        print(f"FAIL: sharpe {m['sharpe']:.4f} < min {args.min_sharpe:.4f}"); ok = False
    if m["maxdd"] > args.maxdd:
        print(f"FAIL: maxdd {m['maxdd']:.4f} > max {args.maxdd:.4f}"); ok = False

    print(f"RESULT: sharpe={m['sharpe']:.4f} maxdd={m['maxdd']:.4f} trades={m['trades']}")
    sys.exit(0 if ok else 1)
