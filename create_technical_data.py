import os
import pandas as pd
import pandas_ta as ta

def creat_technical_data(df_raw, save_cvs=False):
    df = pd.read_csv(df_raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    # 均線
    df.ta.sma(length=7, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=7, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)

    # 趨勢 / 動能
    df.ta.macd(fast=6, slow=13, signal=9, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.rsi(length=7, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)

    # 布林通道
    df.ta.bbands(length=20, std=2, append=True)

    # 動量與成交量相關
    df.ta.obv(append=True)                   # OBV
    df.ta.mfi(length=14, append=True)        # Money Flow Index
    df.ta.stoch(length=14, smooth_k=3, append=True)  # KD 隨機指標
    df.ta.kdj(append=True)                   # KDJ (是 KD 延伸)
    df.ta.uo(append=True)                    # Ultimate Oscillator
    df.ta.trix(length=15, append=True)       # TRIX

    # 報酬率
    df["return"] = df["close"].pct_change()

    df.dropna(inplace=True)

    if save_cvs:
        path = df_raw.replace("full_csv", "technical").replace("full.csv", "technical.csv")
        df.to_csv(path, index=True)

    return df


if __name__ == "__main__":
    coin = "BTCUSDT"

    # 存成 CSV 檔案
    creat_technical_data(f'./full_csv/{coin}_1d_full.csv', save_cvs=True)
    creat_technical_data(f'./full_csv/{coin}_1h_full.csv', save_cvs=True)

    print("✅ 已成功存成 CSV！")