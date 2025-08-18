# backtests/quick_grid_stage2.py
from __future__ import annotations
import csv, subprocess, time
from pathlib import Path
import shutil, sys

ROOT = Path(__file__).resolve().parents[1]
BT = ROOT / "backtests"
RUNS = BT / "runs2"; RUNS.mkdir(parents=True, exist_ok=True)
PY = sys.executable

def run_one(tag, params: dict, minutes=60000):
    out_txt = BT/'out.txt'
    if out_txt.exists():
        out_txt.unlink()
    args = [PY, "-u", "-m", "backtests.run_quick_backtest",
            "--symbol","XAUUSD","--session","ln_ny","--strats","ema,turtle20,turtle55",
            "--minutes", str(minutes)]
    for k,v in params.items():
        args += [f"--{k}", str(v)]
    log = RUNS/f"{tag}.log"
    rc = subprocess.run(args, cwd=ROOT, stdout=open(log,"w",encoding="utf-8"),
                        stderr=subprocess.STDOUT).returncode
    metrics_path = RUNS/f"{tag}.metrics.txt"
    if rc==0 and out_txt.exists() and out_txt.stat().st_size>0:
        shutil.copyfile(out_txt, RUNS/f"{tag}.out.txt")
        r = subprocess.run([PY, str(BT/"print_metrics.py"), str(out_txt)],
                           cwd=ROOT, capture_output=True, text=True)
        metrics_path.write_text(r.stdout or r.stderr, encoding="utf-8")
    else:
        metrics_path.write_text(f"[error] rc={rc}\n", encoding="utf-8")
    return rc

def main():
    s1 = BT/"grid_stage1_pass.csv"
    if not s1.exists():
        print("[!] grid_stage1_pass.csv not found"); sys.exit(2)
    topk = 5
    bases = []
    with open(s1, newline='', encoding='utf-8') as f:
        for i,row in enumerate(csv.DictReader(f)):
            bases.append(row)
            if len(bases)>=topk: break

    rows=[]
    for b in bases:
        base = {"atr_mult": float(b["atr_mult"]),
                "vote": int(float(b["vote"])),
                "cooldown": int(float(b["cooldown"]))}
        for max_layers in [1,2,3]:
            for pyr_step_atr in [0.5,0.8,1.2]:
                params = {**base, "max_layers": max_layers, "pyr_step_atr": pyr_step_atr}
                tag = "s2_"+"_".join(f"{k}{v}" for k,v in params.items())
                print("→", tag); sys.stdout.flush()
                run_one(tag, params, minutes=int(float(b.get("minutes",60000))))
                rows.append(params)

if __name__ == "__main__":
    main()
