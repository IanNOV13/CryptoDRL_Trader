import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
import time
import datetime
from collections import deque
from dotenv import load_dotenv

# 導入你的自定義模塊
# 確保這些模塊的路徑在 PYTHONPATH 中，或者與 predict.py 在同一目錄
from config_loader import ConfigLoader
from create_technical_data import creat_technical_data # 假設此函數能接收 DataFrame
from dqn_lstm_model import build_inception_lstm_dqn_model # 使用你最終的模型構建函數
from utils import notify_discord_webhook, setup_gpu
from get_price import get_all_binance_klines

# --- 全局配置 ---
SYMBOL = "BTCUSDT"
load_dotenv()

def predict_action(model, state_sequence_scaled, action_map):
    """使用單個模型進行預測"""
    # 準備模型輸入
    model_input = state_sequence_scaled.reshape(1, *state_sequence_scaled.shape)
    
    # 預測 Q 值
    q_values = model.predict(model_input, verbose=0)[0]
    
    # 選擇最佳動作
    action_index = np.argmax(q_values)
    
    # 將動作索引映射到描述性文本
    action_ratio = action_map[action_index]
    action_desc = "保持持有或觀望"
    if action_ratio > 0:
        action_desc = f"買入 (倉位 {action_ratio*100:.0f}%)"
    elif action_ratio < 0:
        action_desc = f"賣出 (倉位 {abs(action_ratio)*100:.0f}%)"
        
    return action_index, action_desc, q_values

def get_current_account_state(config, df_latest_day):
    """
    從用戶輸入獲取當前的賬戶狀態。
    在實際應用中，這部分可以替換為從交易所API或數據庫讀取。
    """
    print("\n--- 請輸入當前賬戶狀態 ---")
    
    try:
        # 從 DataFrame 獲取當前價格
        current_price = float(df_latest_day['close'].iloc[-1])
        print(f"(當前參考價格: {current_price:.2f})")

        # 獲取用戶輸入
        initial_balance = 100.0#float(input(f"初始資金 (USDT): "))
        current_balance = 0.0#float(input(f"當前現金餘額 (USDT): "))
        position = 0.0#float(input(f"當前持有幣數量 ({SYMBOL.replace('USDT', '')}): "))
        buy_point = 0#float(input("平均買入成本 (如果空倉則輸入0): "))
        no_action_timer = 0#int(input("距離上次操作已過幾個時間步(天)?: "))

        # 計算衍生狀態
        total_balance = current_balance + position * current_price
        total_balance_ratio = total_balance / initial_balance if initial_balance > 0 else 1.0
        portfolio_crypto_pct = (position * current_price) / total_balance if total_balance > 0 else 0.0
        
        if buy_point > 0:
            relative_buy_point = (current_price - buy_point) / current_price
            signed_sqrt_relative_buy_point = np.sign(relative_buy_point) * np.sqrt(np.abs(relative_buy_point))
        else:
            signed_sqrt_relative_buy_point = 0.0

        return np.array([
            portfolio_crypto_pct,
            total_balance_ratio,
            no_action_timer,
            signed_sqrt_relative_buy_point
        ]), current_price, buy_point

    except ValueError:
        print("[錯誤] 輸入無效，請確保輸入的是數字。將使用默認值。")
        # 返回一組默認值，避免程序崩潰
        return np.array([0.0, 1.0, 0, 0.0]), df_latest_day['close'].iloc[-1], 0.0

def softmax(q_values, temperature=1.0):
    q_values_temp = np.array(q_values) / temperature
    # 為了數值穩定性，減去最大值
    q_values_temp = q_values_temp - np.max(q_values_temp)
    exp_q = np.exp(q_values_temp)
    probabilities = exp_q / np.sum(exp_q)
    return probabilities

def main():
    """主預測循環"""
    print("--- 啟動交易建議腳本 (每小時) ---")
    
    # 1. 加載配置、模型和 Scaler
    try:
        config_obj = ConfigLoader("setting.json")
        config = config_obj.load_settings()
        setup_gpu()

        model_path_balance = config['MODEL_PATH'] + "_BEST_BALANCE"
        model_path_calmar = config['MODEL_PATH'] + "_BEST_CALMAR_RATIO" # 假設你保存了這個模型
        scaler_path = os.path.join(config['MODEL_PATH'], "scaler.joblib")
        action_map = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0, 6: -0.2, 7: -0.4, 8: -0.6, 9: -0.8, 10: -1.0}
        feature_columns = [
            "open", "high", "low", "close", "volume","return", 
            "SMA_50",# "SMA_7", "SMA_20",
            #"EMA_7", "EMA_12", "EMA_26", 
            "ATRr_14", 
            "ADX_14", "DMP_14", "DMN_14",
            "MACD_6_13_9", "MACDh_6_13_9", "MACDs_6_13_9",
            "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
            "RSI_14",# "RSI_7",
            "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0",
            "OBV"
        ]
        sequence_length = config['SEQUENCE_LENGTH']

        print("加載模型和 Scaler...")
        model_balance = tf.keras.models.load_model(model_path_balance)
        model_calmar = tf.keras.models.load_model(model_path_calmar)
        scaler = joblib.load(scaler_path)
        print("模型和 Scaler 加載成功！")
    except Exception as e:
        print(f"[致命錯誤] 加載配置、模型或 Scaler 失敗: {e}")
        return

    # 2. 主循環，每小時執行一次
    while True:
        try:
            # --- 數據獲取 ---
            print("\n" + "="*50)
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 開始新一輪預測...")
            # 獲取包含今天這根不完整日K線在內的所有歷史日線數據
            df_daily_full = get_all_binance_klines(symbol=SYMBOL, interval="1d")
            print(f"已獲取 {len(df_daily_full)} 條日線數據，最新一條為 {df_daily_full['timestamp'].iloc[-1].date()}")

            # --- 狀態準備 ---
            # 1. 計算技術指標
            df_with_indicators = creat_technical_data(df_daily_full.copy())
            print(df_with_indicators)
            
            # 2. 準備市場狀態序列
            market_state_sequence = df_with_indicators[feature_columns].tail(sequence_length).values
            
            if len(market_state_sequence) < sequence_length:
                raise ValueError(f"數據不足以構成長度為 {sequence_length} 的序列。")
                
            # 3. 獲取用戶輸入的當前賬戶狀態
            account_state, current_price, current_buy_point = get_current_account_state(config, df_daily_full)
            
            # 4. 組合完整的狀態序列
            account_state_sequence = np.tile(account_state, (sequence_length, 1))
            full_state_sequence = np.hstack([market_state_sequence, account_state_sequence])
            
            # 5. 歸一化
            state_sequence_scaled = scaler.transform(full_state_sequence)

            # --- 模型預測 ---
            print("\n--- 模型預測結果 ---")
            # 最高餘額模型
            bal_action_idx, bal_action_desc, bal_q_values = predict_action(
                model_balance, state_sequence_scaled, action_map
            )
            bal_probs = softmax(bal_q_values)
            # 最高卡爾瑪模型
            cal_action_idx, cal_action_desc, cal_q_values = predict_action(
                model_calmar, state_sequence_scaled, action_map
            )
            cal_probs = softmax(cal_q_values)

            # --- 綜合建議與通知 ---
            report = f"""
🔔 **交易建議** ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}) 🔔
-----------------------------------
- **當前 {SYMBOL} 價格**: {current_price:.2f}
- **你的持倉成本**: {current_buy_point:.2f}
-----------------------------------
📈 **最高餘額模型 (追求高回報)**:
    - **建議操作**: **{bal_action_desc}**
    - 動作索引: {bal_action_idx}
    - Q值: {[f'{q:.2f}' for q in bal_q_values]}
    - 信心:{[f'{p*100:.2f}%' for p in bal_probs]}

🛡️ **最高卡爾瑪模型 (注重風險控制)**:
    - **建議操作**: **{cal_action_desc}**
    - 動作索引: {cal_action_idx}
    - Q值: {[f'{q:.2f}' for q in cal_q_values]}
    - 信心:{[f'{p*100:.2f}%' for p in cal_probs]}
-----------------------------------
            """
            print(report)
            notify_discord_webhook(report,webhook_url=os.getenv("DISCORD_WEBHOOK_2"))

        except Exception as e:
            error_message = f"[錯誤] 預測循環中發生錯誤: {e}"
            import traceback
            traceback.print_exc()
            notify_discord_webhook(error_message)

        # --- 等待下一小時 ---
        print("="*50)
        now = datetime.datetime.now()
        seconds_until_next_hour = (60 - now.minute - 1) * 60 + (60 - now.second)
        for i in range(seconds_until_next_hour):
            print(f"下一次預測還有{seconds_until_next_hour - i}... (按 Ctrl+C 退出)",end="\r")
            time.sleep(1)
        print("")

if __name__ == "__main__":
    main()