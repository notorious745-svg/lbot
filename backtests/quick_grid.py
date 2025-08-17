# backtests/quick_grid.py  (with live progress + ETA)
import csv, itertools, subprocess, sys, time, os, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BT = ["python","-u","-m","backtests.run_quick_backtest"]
PR = ["python","-u", str(ROOT / "backtests" / "print_metrics.py")]

def run_one(minutes, symbol, session, strats, atr_mult, vote, cooldown, max_layers, pyr_step):
    out = ROOT / "backtests" / "out.txt"
    met = ROOT / "metrics.txt"
    if out.exists(): out.unlink(missing_ok=True)
    if met.exists(): met.unlink(missing_ok=True)

    args = BT + [
        "--minutes", str(minutes),
        "--symbol", symbol,
        "--session", session,
        "--strats", ",".join(strats),
        "--atr_mult", str(atr_mult),
        "--vote", str(vote),
        "--cooldown", str(cooldown),
        "--max_layers", str(max_layers),
        "--pyr_step_atr", str(pyr_step),
    ]
    subprocess.run(args, check=True)
    with open(met, "w", encoding="utf-8") as f:
        subprocess.run(PR + [str(out)], check=True, stdout=f)

    text = met.read_text(encoding="utf-8", errors="ignore")
    def f(key):
        m = re.search(rf"{key}\s*=\s*([-\d\.]+)", text, re.I)
        return float(m.group(1)) if m else float("nan")
    return {
        "sharpe": f("sharpe"),
        "maxdd": f("maxdd"),
        "trades": f("trades"),
        "session": session,
        "strats": ",".join(strats),
        "atr_mult": atr_mult,
        "vote": vote,
        "cooldown": cooldown,
        "max_layers": max_layers,
        "pyr_step_atr": pyr_step,
    }

def fmt_eta(sec):
    if sec is None or sec != sec: return "?"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

def main(minutes=60000, symbol="XAUUSD"):
    SESS = ["ln_ny", "all"]
    STRS = [
        ("ema","turtle20","turtle55"),
        ("ema","turtle20"),
        ("turtle20","turtle55"),
    ]
    ATRS = [2.5, 3.0, 3.5]
    VOTE = [1, 2]
    COOL = [0, 4, 8]
    LAYS = [0, 1]
    PYRS = [1.0, 1.2]

    grid = list(itertools.product(SESS, STRS, ATRS, VOTE, COOL, LAYS, PYRS))
    total = len(grid)
    results = []
    t0 = time.time()
    prog_file = ROOT / "backtests" / "progress.json"
    log_file = ROOT / "backtests" / "grid_log.txt"
    log = open(log_file, "a", encoding="utf-8")

    for i,(session,strats,atr,vote,cool,layers,pyr) in enumerate(grid, 1):
        t1 = time.time()
        try:
            row = run_one(minutes, symbol, session, strats, atr, vote, cool, layers, pyr)
            results.append(row)
            ok = True
            msg = (f"[{i}/{total}] sharpe={row['sharpe']:.3f} dd={row['maxdd']:.3f} "
                   f"trades={row['trades']}  {session} {strats} atr{atr} vote{vote} cd{cool} L{layers} step{pyr}")
        except subprocess.CalledProcessError as e:
            ok = False
            msg = f"[{i}/{total}] FAILED {session} {strats} atr{atr} vote{vote} cd{cool} L{layers} step{pyr}: {e}"

        # progress/eta
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0.0
        rem = (total - i) / rate if rate > 0 else None
        pct = round(100.0 * i / total, 1)
        line = f"{msg}  | {pct:.1f}%  ETA {fmt_eta(rem)}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()
        prog_file.write_text(json.dumps({
            "total": total, "done": i, "percent": pct,
            "elapsed_sec": int(elapsed), "eta_sec": int(rem) if rem else None,
            "last_msg": msg
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    outcsv = ROOT / "backtests" / "grid_results.csv"
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    with open(outcsv,"w",newline="",encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader(); wr.writerows(results)

    print(f"\nSaved -> {outcsv}  ({len(results)} rows, took {time.time()-t0:.1f}s)")
    log.write(f"\nSaved -> {outcsv}\n"); log.close()

    passed = [r for r in results if r["maxdd"] <= 0.25 and r["trades"] >= 100]
    passed.sort(key=lambda r: (-r["sharpe"], r["maxdd"]))
    if passed:
        best = passed[0]
        bestfile = ROOT / "backtests" / "best_config.txt"
        bestfile.write_text("\n".join(f"{k}={v}" for k,v in best.items()), encoding="utf-8")
        print("Best:", best); print(f"Saved -> {bestfile}")

if __name__ == "__main__":
    mins = int(sys.argv[1]) if len(sys.argv)>1 else 60000
    sym  = sys.argv[2] if len(sys.argv)>2 else "XAUUSD"
    main(mins, sym)
