import requests
import pandas as pd
import time

def get_all_binance_klines(symbol="BTCUSDT", interval="1d", end_time=None):
    """ 取得幣安所有歷史 K 線數據 """
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    limit = 1000  # 幣安 API 每次最多回傳 1000 筆
    try:
        while len(all_data) < 100000:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "endTime": end_time
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if not data:
                break  # 如果沒有新數據，結束迴圈

            all_data.extend(data)
            end_time = data[0][0] - 1  # 設定下一次請求的起始時間
            
            print(f"📊 已獲取 {len(all_data)} 筆 {symbol} {interval} K 線數據...")

            time.sleep(0.5)  # 避免 API 請求過快被限制
    except:
        pass
        
    # 轉換為 DataFrame
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # 轉換時間格式
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.sort_values(by="timestamp", ascending=True, inplace=True)

    # 選擇需要的欄位
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    return df

if __name__ == "__main__":
    coin = "BTCUSDT"
    # 取得 BTC/USDT 的完整歷史 K 線數據
    df_daily = get_all_binance_klines(symbol=coin, interval="1d")
    #df_hourly = get_all_binance_klines(symbol=coin, interval="1h")

    # 存成 CSV 檔案
    df_daily.to_csv(f"./full_csv/{coin}_1d_full.csv", index=False)
    #df_hourly.to_csv(f"./full_csv/{coin}_1h_full.csv", index=False)

    print("✅ 已成功存成 CSV！")
