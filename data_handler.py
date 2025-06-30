# data_handler.py

import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler # 引入兩種 Scaler
from trading_env import CryptoTradingEnv # 假設 trading_env.py 在同一目錄或 PYTHONPATH 中
import numpy as np
from create_technical_data import creat_technical_data
import gc

# --- 主函數：準備數據和環境 ---
def prepare_data_and_env(config, symbol="BTCUSDT", interval="1d"):
    """
    加載數據、計算技術指標、創建訓練和測試環境，並處理 Scaler。
    返回: train_env, test_env, scaler (擬合好的)
    """
    print("--- 開始準備數據和環境 ---")

    # 1. 數據路徑和文件名
    raw_data_filename = f"{symbol}_{interval}_full.csv" # 假設這是你原始數據的文件名格式
    raw_data_path = os.path.join(config.get("RAW_DATA_DIR", "./full_csv"), raw_data_filename) # 從配置或默認路徑加載

    technical_data_dir = config.get("TECHNICAL_DATA_DIR", "./technical")
    technical_data_filename = f"{symbol}_{interval}_technical.csv"
    technical_data_path = os.path.join(technical_data_dir, technical_data_filename)

    os.makedirs(technical_data_dir, exist_ok=True)

    # 2. 檢查技術指標文件是否存在，否則創建它
    if not os.path.exists(technical_data_path) or config.get("FORCE_RECALCULATE_INDICATORS", False):
        print(f"技術指標文件 {technical_data_path} 未找到或強制重新計算...")
        try:
            print(f"正在從 {raw_data_path} 加載原始數據...")
            df_raw = pd.read_csv(raw_data_path)
            if "timestamp" not in df_raw.columns: # 確保有 timestamp 列
                # 假設第一列是時間戳（如果格式不同需要調整）
                df_raw.rename(columns={df_raw.columns[0]: "timestamp"}, inplace=True)

            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
            df_raw.set_index("timestamp", inplace=True)

            print("正在計算技術指標...")
            df_technical = creat_technical_data(df_raw)
            df_technical.to_csv(technical_data_path) # 保存索引 (timestamp)
            print(f"技術指標計算完成，已保存到 {technical_data_path}")
        except FileNotFoundError:
            print(f"[錯誤] 原始數據文件 {raw_data_path} 未找到！請確保文件存在。")
            exit()
        except Exception as e:
            print(f"[錯誤] 處理原始數據或計算指標時失敗: {e}")
            exit()
    else:
        print(f"從已有的 {technical_data_path} 加載技術指標數據。")

    # 3. 創建環境
    #    注意：CryptoTradingEnv 的 __init__ 應該只接收包含所有必要列的數據文件路徑
    #    它內部會處理數據加載和劃分
    print("創建訓練和測試環境...")
    train_env = CryptoTradingEnv(
        df=technical_data_path,
        data_split='train',
        train_ratio=config.get("TRAIN_RATIO", 0.8),
        initial_balance=config.get("INITIAL_BALANCE", 500)
    )
    test_env = CryptoTradingEnv(
        df=technical_data_path,
        data_split='test',
        train_ratio=config.get("TRAIN_RATIO", 0.8), # 保持與訓練環境一致的分割比例
        initial_balance=config.get("INITIAL_BALANCE", 500)
    )

    # 4. 處理 Scaler (加載或擬合)
    scaler_name = "scaler.joblib"
    scaler_path = os.path.join(config["MODEL_PATH"], scaler_name)
    scaler_loaded = False
    scaler_to_use = None

    if os.path.exists(scaler_path) and config.get("LOAD_SCALER_IF_EXISTS", True):
        try:
            scaler_to_use = joblib.load(scaler_path)
            print(f"Scaler 加載自: {scaler_path}")
            scaler_loaded = True
        except Exception as e:
            print(f"加載 Scaler 失敗 ({scaler_path}): {e}. 將重新收集數據並擬合。")

    if not scaler_loaded:
        print("開始收集狀態數據用於擬合 Scaler...")
        initial_states_for_scaler = []
        # 使用一個臨時的訓練環境來收集數據
        temp_env_for_scaler = CryptoTradingEnv(
            df=technical_data_path,
            data_split='train', # 必須使用訓練數據擬合
            train_ratio=config.get("TRAIN_RATIO", 0.8),
            initial_balance=config.get("INITIAL_BALANCE", 500)
        )
        # 選擇 Scaler 類型
        scaler_type = config.get("SCALER_TYPE", "StandardScaler") # 從配置讀取，默認 StandardScaler
        if scaler_type == "RobustScaler":
            print("使用 RobustScaler。")
            scaler_to_use = RobustScaler()
        else:
            print("使用 StandardScaler。")
            scaler_to_use = StandardScaler()

        # 收集數據
        temp_state_unscaled = temp_env_for_scaler.get_current_unscaled_state_vector()
        initial_states_for_scaler.append(temp_state_unscaled)
        done = False
        step_count = 0
        # 確保 temp_env_for_scaler.data 不是空的
        if temp_env_for_scaler.data is None or len(temp_env_for_scaler.data) == 0:
            print("[錯誤] 臨時環境中沒有數據用於收集Scaler樣本！")
            exit()
        max_collect_steps = len(temp_env_for_scaler.data) - 1

        while not done and step_count < max_collect_steps:
            action = temp_env_for_scaler.action_space.sample()
            _, _, done, _ = temp_env_for_scaler.step(action)
            # 確保在 step 之後，current_step 沒有超出 data 範圍太多
            if temp_env_for_scaler.current_step < len(temp_env_for_scaler.data):
                 next_state_unscaled = temp_env_for_scaler.get_current_unscaled_state_vector()
                 initial_states_for_scaler.append(next_state_unscaled)
            step_count += 1

        print(f"總共收集了 {len(initial_states_for_scaler)} 個狀態用於擬合 Scaler。")
        scaler_data_np = np.array(initial_states_for_scaler)

        if scaler_data_np.shape[0] > 0 and scaler_data_np.shape[1] == train_env.observation_space.shape[0]:
            print(f"正在使用 {scaler_type} 擬合 Scaler...")
            scaler_to_use.fit(scaler_data_np) # 用收集的數據擬合選擇的 scaler
            print("Scaler 擬合完成。")
            try:
                # 確保模型目錄存在 (SCALER_PATH 通常在模型目錄下)
                model_dir = os.path.dirname(scaler_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                joblib.dump(scaler_to_use, scaler_path)
                print(f"Scaler 已保存到: {scaler_path}")
            except Exception as e:
                print(f"[錯誤] 保存 Scaler 失敗: {e}")
        else:
            print(f"[錯誤] 收集到的數據維度不正確或沒有收集到數據！Shape: {scaler_data_np.shape}, Expected columns: {train_env.observation_space.shape[0]}")
            exit()
        del temp_env_for_scaler, initial_states_for_scaler, scaler_data_np # 清理
        gc.collect()

    # 將最終確定使用的 scaler 應用到環境
    if scaler_to_use:
        train_env.scaler = scaler_to_use
        test_env.scaler = scaler_to_use # 測試環境使用相同的 scaler
        train_env._scaler_fitted = True
        test_env._scaler_fitted = True
        print("Scaler 已成功配置到訓練和測試環境。")
    else:
        print("[錯誤] 未能成功加載或擬合 Scaler！")
        exit()

    print("--- 數據和環境準備完成 ---")
    return train_env, test_env, scaler_to_use # 返回擬合好的 scaler 實例