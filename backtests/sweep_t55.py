from __future__ import annotations
import itertools, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = [sys.executable, str(ROOT/"backtests"/"run_quick_backtest.py")]
MET = [sys.executable, str(ROOT/"backtests"/"print_metrics.py"), str(ROOT/"backtests"/"out.txt")]

def run_once(args: list[str]) -> dict:
    # รันตัวแบ็กเทสต์
    subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # อ่าน metrics
    out = subprocess.run(MET, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    f = lambda k: float(re.search(fr"{k}=([-\d\.]+)", out).group(1))
    return {"sharpe": f("sharpe"), "maxdd": f("maxdd"), "trades": int(f("trades"))}

def main():
    minutes   = 60000
    session   = "ln_ny"
    strats    = "turtle55"
    votes     = [1, 2]
    cooldowns = [4, 6, 8, 12]
    atr_mults = [1.5, 2.0, 3.0, 4.0]

    rows = []
    for v, cd, am in itertools.product(votes, cooldowns, atr_mults):
        args = RUN + ["--minutes", str(minutes), "--session", session,
                      "--strats", strats, "--vote", str(v),
                      "--cooldown", str(cd), "--max_layers", "1",
                      "--atr_mult", str(am)]
        r = run_once(args)
        r.update({"vote": v, "cooldown": cd, "atr_mult": am})
        rows.append(r)
        print(f"[done] vote={v} cd={cd} atr={am}  -> Sharpe={r['sharpe']:.3f} DD={r['maxdd']:.3f} Trades={r['trades']}")

    # จัดอันดับแล้วพิมพ์ Top-5
    rows.sort(key=lambda x: (x["sharpe"], -x["maxdd"]), reverse=True)
    print("\nTOP 5")
    for i, r in enumerate(rows[:5], 1):
        print(f"{i:>2}) vote={r['vote']} cd={r['cooldown']} atr={r['atr_mult']}  | Sharpe={r['sharpe']:.3f}  DD={r['maxdd']:.3f}  Trades={r['trades']}")

if __name__ == "__main__":
    main()
