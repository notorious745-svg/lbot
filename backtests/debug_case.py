# backtests/debug_case.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

# ปรับชื่อ/ที่อยู่ให้ตรงกับโปรเจกต์คุณ
DATA = Path('data/XAUUSD_15m_clean.csv')

def main():
    df = pd.read_csv(DATA)
    print('rows_all =', len(df))
    print('cols =', list(df.columns)[:12], '...')
    # เคสทั่วไป: ต้องมีคอลัมน์ time/open/high/low/close/volume เป็นต้น
    # ลองดูช่วงเวลา
    if 'time' in df.columns:
        print('time_min =', df['time'].iloc[0], 'time_max =', df['time'].iloc[-1])

    # ชิม ๆ สัญญาณง่าย ๆ: EMA 10/20 cross + Turtle 20/55 break
    if {'close'}.issubset(df.columns):
        for p in (10,20,50):
            df[f'ema{p}'] = df['close'].ewm(span=p, min_periods=p).mean()

        df['ema_long']  = (df['ema10'] > df['ema20']) & (df['ema20'] > df['ema50'])
        df['ema_short'] = (df['ema10'] < df['ema20']) & (df['ema20'] < df['ema50'])

        print('ema_long signals =', int(df['ema_long'].sum()))
        print('ema_short signals =', int(df['ema_short'].sum()))

        # Turtle breakout
        w1, w2 = 20, 55
        df['hh20'] = df['close'].rolling(w1).max().shift(1)
        df['ll20'] = df['close'].rolling(w1).min().shift(1)
        df['hh55'] = df['close'].rolling(w2).max().shift(1)
        df['ll55'] = df['close'].rolling(w2).min().shift(1)

        df['t20_long']  = df['close'] > df['hh20']
        df['t20_short'] = df['close'] < df['ll20']
        df['t55_long']  = df['close'] > df['hh55']
        df['t55_short'] = df['close'] < df['ll55']

        print('t20_long=', int(df['t20_long'].sum()),
              't20_short=', int(df['t20_short'].sum()),
              't55_long=', int(df['t55_long'].sum()),
              't55_short=', int(df['t55_short'].sum()))

if __name__ == '__main__':
    main()
