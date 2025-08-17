# backtests/quick_grid.py
from __future__ import annotations
import csv, itertools, os, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]   # repo root (…/lbot)
BT_DIR = ROOT / "backtests"
DATA = ROOT / "data" / "XAUUSD_15m_clean.csv"
RUNS_DIR = BT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

PY = sys.executable

# ---------- helpers ----------
def run_cmd(args: List[str], log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8", newline="") as logf:
        p = subprocess.run(args, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
        return p.returncode

METRICS_RE = {
    "sharpe": re.compile(r"Sharpe\s*[:=]\s*([-+]?\d*\.?\d+)"),
    "maxdd":  re.compile(r"MaxDD\s*[:=]\s*([-+]?\d*\.?\d+)"),
    "trades": re.compile(r"Trades?\s*[:=]\s*(\d+)"),
}

def parse_metrics(text: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    sh = md = None
    td = None
    m = METRICS_RE["sharpe"].search(text)
    if m: sh = float(m.group(1))
    m = METRICS_RE["maxdd"].search(text)
    if m: md = float(m.group(1))
    m = METRICS_RE["trades"].search(text)
    if m: td = int(m.group(1))
    return sh, md, td

def minutes_to_days(minutes: int) -> float:
    return minutes / 1440.0

def safe_remove(p: Path):
    try:
        p.unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass

def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

# ---------- core run ----------
def run_one(tag: str, base_args: Dict[str, str|int|float], minutes: int) -> Dict[str, object]:
    """Run a single backtest case, return dict with params + metrics."""
    out_txt = BT_DIR / "out.txt"
    run_log = RUNS_DIR / f"{tag}.log"

    # cleanup
    safe_remove(out_txt)

    args = [
        PY, "-u", "-m", "backtests.run_quick_backtest",
        "--symbol", "XAUUSD",
        "--session", "ln_ny",
        "--strats", "ema,turtle20,turtle55",
        "--minutes", str(minutes),
    ]

    # attach params
    for k, v in base_args.items():
        args += [f"--{k}", str(v)]

    t0 = time.time()
    rc = run_cmd(args, run_log)
    dur = time.time() - t0

    # metrics
    metrics_txt = RUNS_DIR / f"{tag}.metrics.txt"
    if rc == 0 and out_txt.exists() and out_txt.stat().st_size > 0:
        # copy raw out file for archive
        archived = RUNS_DIR / f"{tag}.out.txt"
        shutil.copyfile(out_txt, archived)

        # call print_metrics.py
        r = subprocess.run(
            [PY, str(BT_DIR / "print_metrics.py"), str(out_txt)],
            capture_output=True, text=True, cwd=ROOT
        )
        metrics_txt.write_text(r.stdout or r.stderr, encoding="utf-8")
        sh, md, td = parse_metrics(metrics_txt.read_text(encoding="utf-8"))
    else:
        sh = md = None
        td = None
        metrics_txt.write_text(f"[error] rc={rc}, out.txt missing/empty\n", encoding="utf-8")

    days = minutes_to_days(minutes)
    tpd = (td / days) if (td is not None and days > 0) else None

    row = {
        "tag": tag, "rc": rc, "seconds": round(dur, 2),
        **{k: base_args[k] for k in base_args},
        "minutes": minutes,
        "sharpe": sh, "maxdd": md, "trades": td, "trades_per_day": tpd,
        "log": str(run_log.relative_to(ROOT)),
        "out": str((RUNS_DIR / f"{tag}.out.txt").relative_to(ROOT)),
    }
    return row

def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

# ---------- Stage 1: coarse baseline (no pyramiding) ----------
def stage1(minutes: int = 60000, outfile: Path | None = None) -> Path:
    if outfile is None:
        outfile = BT_DIR / "grid_stage1.csv"

    grid = {
        "atr_mult":  [2.0, 2.5, 3.0, 3.5],
        "vote":      [1, 2, 3],
        "cooldown":  [0, 6, 12],
        "max_layers":[0],                   # disable pyramiding
    }
    combos = list(itertools.product(*grid.values()))
    name_order = list(grid.keys())

    rows = []
    for i, values in enumerate(combos, 1):
        params = {k: v for k, v in zip(name_order, values)}
        tag = f"s1_{i:03d}_" + "_".join(f"{k}{v}" for k, v in params.items())
        print(f"[Stage1] {i}/{len(combos)} → {tag}")
        row = run_one(tag, params, minutes=minutes)
        rows.append(row)

    # sort by sharpe desc, then trades/day desc
    rows_sorted = sorted(
        rows,
        key=lambda r: (-(r["sharpe"] if r["sharpe"] is not None else -1e9),
                       -(r["trades_per_day"] if r["trades_per_day"] is not None else -1e9))
    )
    write_csv(outfile, rows_sorted)

    # also write a “passed filter” CSV (≥3 trades/day)
    passed = [r for r in rows_sorted if (r["trades_per_day"] or 0) >= 3]
    write_csv(BT_DIR / "grid_stage1_pass.csv", passed)
    return outfile

# ---------- Stage 2: pyramiding around top-K from Stage 1 ----------
def stage2(topk: int = 5, minutes: int = 60000, infile: Optional[Path] = None, outfile: Optional[Path] = None) -> Path:
    infile = infile or (BT_DIR / "grid_stage1_pass.csv")
    outfile = outfile or (BT_DIR / "grid_stage2.csv")

    if not infile.exists():
        raise FileNotFoundError(f"Stage1 results not found: {infile}")

    # read topK
    with open(infile, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_rows = [row for row in reader]

    base_rows = base_rows[:topk]
    rows = []
    for b in base_rows:
        base_params = {
            "atr_mult": float(b["atr_mult"]),
            "vote": int(float(b["vote"])),
            "cooldown": int(float(b["cooldown"])),
        }
        for max_layers in [1, 2, 3]:
            for pyr_step_atr in [0.5, 0.8, 1.2]:
                params = {**base_params, "max_layers": max_layers, "pyr_step_atr": pyr_step_atr}
                tag = ("s2_" + "_".join(
                    f"{k}{v}" for k, v in params.items()
                ))
                print(f"[Stage2] → {tag}")
                row = run_one(tag, params, minutes=minutes)
                # keep baseline info for traceability
                row["base_tag"] = b["tag"]
                row["base_sharpe"] = b["sharpe"]
                row["base_trades_per_day"] = b["trades_per_day"]
                rows.append(row)

    # rank by sharpe then trades/day
    rows_sorted = sorted(
        rows,
        key=lambda r: (-(r["sharpe"] if r["sharpe"] is not None else -1e9),
                       -(r["trades_per_day"] if r["trades_per_day"] is not None else -1e9))
    )
    write_csv(outfile, rows_sorted)
    return outfile

def main():
    # sanity check: price file present
    if not DATA.exists():
        print(f"[!] Missing data file: {DATA}")
        sys.exit(2)

    # Stage 1
    print("=== Stage 1: Coarse grid (no pyramiding) ===")
    s1_csv = stage1(minutes=60000)
    print(f"[OK] Stage1 → {s1_csv.relative_to(ROOT)}")
    # Stage 2
    print("=== Stage 2: Pyramiding around top-K from Stage1 ===")
    s2_csv = stage2(topk=5, minutes=60000)
    print(f"[OK] Stage2 → {s2_csv.relative_to(ROOT)}")
    print("Done.")

if __name__ == "__main__":
    main()
