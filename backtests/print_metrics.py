from __future__ import annotations
import re, sys, numpy as np

def _first_num(patterns, text, default=None):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try: return float(m.group(1))
            except: pass
    return default

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("sharpe=0\nmaxdd=1\ntrades=0"); sys.exit(0)
    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        print("sharpe=0\nmaxdd=1\ntrades=0"); sys.exit(0)

    # รองรับทั้งรูปแบบใหม่ (KEY=VALUE) และข้อความเก่า ๆ
    sharpe = _first_num([r"SHARPE\s*=\s*([\-+]?\d+(\.\d+)?)",
                         r"sharpe[:\s]\s*([\-+]?\d+(\.\d+)?)",
                         r"sharpe≈\s*([\-+]?\d+(\.\d+)?)"], txt, default=0.0)
    maxdd  = _first_num([r"MAXDD\s*=\s*([\-+]?\d+(\.\d+)?)",
                         r"maxdd[:\s]\s*([\-+]?\d+(\.\d+)?)",
                         r"drawdown[:\s]\s*([\-+]?\d+(\.\d+)?)"], txt, default=1.0)
    trades = _first_num([r"TRADES\s*=\s*(\d+)",
                         r"trades[:\s]\s*(\d+)"], txt, default=0.0)

    # พิมพ์เป็น metrics.txt ที่ enforce_gate จะอ่าน
    print(f"sharpe={sharpe:.6f}")
    print(f"maxdd={maxdd:.6f}")
    print(f"trades={int(trades)}")
