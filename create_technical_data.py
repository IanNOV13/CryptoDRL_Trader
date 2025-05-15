import os
import pandas as pd
import pandas_ta as ta

def creat_technical_data(symbol="BTCUSDT", interval="1d"):
    df = pd.read_csv(f"./full_csv/{symbol}_{interval}_full.csv")
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

    output_dir = "./technical"
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f"{output_dir}/{symbol}_{interval}_technical.csv"
    df.to_csv(output_filepath)
    print(f"✅ 技術指標計算完成 (使用 pandas-ta)，已存成 {output_filepath}！")


if __name__ == "__main__":
    coin = "DOGEUSDT"

    # 存成 CSV 檔案
    creat_technical_data(coin, '1d')
    creat_technical_data(coin, '1h')

    print("✅ 已成功存成 CSV！")