# backtests/print_metrics.py
# รองรับ out.txt ไม่มี header, 9 คอลัมน์:
# idx, close, want_long, want_short, enter_long, enter_short, exit_pos, pos, equity
from __future__ import annotations
import sys, numpy as np, pandas as pd

def main(path: str):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 9:
        # ฟอร์แมตอื่น → ลองเดาง่ายๆ: ใช้คอลัมน์สุดท้ายเป็น equity แล้วนับ pos เปลี่ยนสัญญาณ
        eq = df.iloc[:, -1].astype(float).values
        pos = np.sign(np.diff(df.iloc[:, -2].astype(float).values, prepend=0))
        trades = int(np.count_nonzero(pos[1:] != pos[:-1] ))
    else:
        enter_long  = df.iloc[:,4].astype(float)
        enter_short = df.iloc[:,5].astype(float)
        exit_pos    = df.iloc[:,6].astype(float)
        pos         = df.iloc[:,7].astype(float)
        eq          = df.iloc[:,8].astype(float).values
        trades = int(exit_pos.sum())
        if trades == 0:
            trades = int((enter_long + enter_short).sum())

    # คำนวณ Sharpe จากรีเทิร์นต่อแท่ง (15 นาที → 96 แท่ง/วัน, 252 วัน/ปี)
    r = np.diff(eq) / np.where(eq[:-1]==0, 1.0, eq[:-1])
    ann = np.sqrt(96*252.0)
    sharpe = 0.0 if r.std()==0 else float(r.mean()/r.std()*ann)

    # Max drawdown จาก equity (หน่วยเดียวกับ equity)
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.max(peak - eq)) if len(eq) else 0.0

    print(f"sharpe={sharpe:.6f}")
    print(f"maxdd={maxdd:.6f}")
    print(f"trades={trades}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backtests/print_metrics.py backtests/out.txt")
        sys.exit(2)
    main(sys.argv[1])
