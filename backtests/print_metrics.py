# backtests/print_metrics.py
from __future__ import annotations
import sys, re

def read_text_any(path: str) -> str:
    # รองรับไฟล์จาก PowerShell (UTF-16 LE), UTF-8 (มี/ไม่มี BOM) และ fallback อื่นๆ
    encs = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252"]
    with open(path, "rb") as f:
        b = f.read()
    for enc in encs:
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="ignore")

def _first_num(patterns, text, default=None):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    return default

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("sharpe=0\nmaxdd=1\ntrades=0"); sys.exit(0)

    txt = read_text_any(sys.argv[1])

    sharpe = _first_num([r"SHARPE\s*=\s*([\-+]?\d+(\.\d+)?)",
                         r"sharpe[:\s]\s*([\-+]?\d+(\.\d+)?)",
                         r"sharpe≈\s*([\-+]?\d+(\.\d+)?)"], txt, default=0.0)
    maxdd  = _first_num([r"MAXDD\s*=\s*([\-+]?\d+(\.\d+)?)",
                         r"maxdd[:\s]\s*([\-+]?\d+(\.\d+)?)",
                         r"drawdown[:\s]\s*([\-+]?\d+(\.\d+)?)"], txt, default=1.0)
    trades = _first_num([r"TRADES\s*=\s*(\d+)",
                         r"trades[:\s]\s*(\d+)"], txt, default=0.0)

    print(f"sharpe={sharpe:.6f}")
    print(f"maxdd={maxdd:.6f}")
    print(f"trades={int(trades)}")
