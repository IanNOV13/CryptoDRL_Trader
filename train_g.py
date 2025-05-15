import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import mixed_precision # type: ignore
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from collections import deque
from trading_env import CryptoTradingEnv # 假設這個 class 裡有 self.df 和 self.data (用於 state)
from dqn_gru_model import build_inception_gru_dqn_model
import gc
import pandas as pd # 引入 pandas 方便處理 DataFrame
from dotenv import load_dotenv
from linebot.v3.messaging import (
    ApiClient, # API 客戶端
    Configuration, # 配置對象
    MessagingApi, # 主要的 API 接口
    PushMessageRequest, # 推送消息的請求體
    TextMessage # 文本消息對象 (注意路徑不同了)
)
import os
import psutil
import requests
import joblib
import datetime
import signal
import sys
import json

# --- 載入設定檔 ---
with open("setting.json", "r") as f:
    setting = json.load(f)

# --- 參數設定 ---
LOAD_MODEL_PATH = setting["LOAD_MODEL_PATH"]
MODEL_PATH = setting["MODEL_PATH"] #模型名稱
SCALER_PATH = os.path.join(MODEL_PATH, "scaler.joblib")
EPISODES = setting["EPISODES"]  #總訓練回合
BATCH_SIZE = setting["BATCH_SIZE"]
GAMMA = setting["GAMMA"]
INITIAL_EPSILON = setting["INITIAL_EPSILON"]  # 探索率週期的起始值
MIN_EPSILON_CYCLE = setting["MIN_EPSILON_CYCLE"] # 探索率在週期結束時的最小值 (略高於全局 EPSILON_MIN)
EPSILON_CYCLE_LENGTH = setting["EPSILON_CYCLE_LENGTH"] # 多少個回合一個探索週期 (可以基於回合數)
EPSILON_MIN = setting["EPSILON_MIN"] #最小探索率
MEMORY_SIZE = setting["MEMORY_SIZE"] # 經驗池大小可以大一些，增加樣本多樣性
INITIAL_LR = setting["INITIAL_LR"] # 初始學習率 (建議 RL 中用稍小的值開始)0.0001
LR_DROP_RATE = setting["LR_DROP_RATE"] # 最終學習率比例
EPOCHS_DROP_RL = setting["EPOCHS_DROP_RL"] # 退火週期的總訓練步數 (需要估算或設定)
ALPHA_LR = setting["ALPHA_LR"]      # 最終最小學習率
TIME_RANGE = setting["TIME_RANGE"] # 預設繪圖顯示的時間窗口
NUM_PLOTS = setting["NUM_PLOTS"]     # 窗口分割成幾張圖片
START_EPISODE = setting["START_EPISODE"]
BEST_STEPS = setting["BEST_STEPS"]
BEST_REWARD = setting["BEST_REWARD"] if setting["BEST_REWARD"] != None else -np.inf
BEST_BALANCE = setting["BEST_BALANCE"] if setting["BEST_BALANCE"] != None else -np.inf
SEQUENCE_LENGTH = setting["SEQUENCE_LENGTH"] #時間步長
MEMORY_LIMIT_GB = setting["MEMORY_LIMIT_GB"]
USE_LINE_BOT = setting["USE_LINE_BOT"]

# 忽略 INFO 和 WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 載入環境變數
load_dotenv()

# --- GPU 設定 ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Mixed precision policy set to:', mixed_precision.global_policy())

# --- 建立Ctrl+C 或系統送出終止訊號的保存動作 ---
def save_on_exit(signal_received, frame):
    print("中斷訊號捕捉到，儲存模型與訓練進度中...")
    model.save(MODEL_PATH, save_format='tf')
    target_model.save(MODEL_PATH + "_target", save_format='tf')
    # 儲存訓練進度
    progress = {
        "START_EPISODE": episode + 1,
        "BEST_REWARD": float(BEST_REWARD),
        "BEST_BALANCE": float(BEST_BALANCE),
        "EPSILON": float(EPSILON), # 當前的 Epsilon
        "BEST_STEPS": BEST_STEPS,
        "current_lr": float(current_lr) if 'current_lr' in globals() and current_lr is not None else INITIAL_LR, # 當前學習率
        "global_step_count": global_step_count if 'global_step_count' in globals() else 0
    }

    # 應該是更新 setting 字典，然後寫回整個 setting 字典
    existing_settings = {}
    try:
        with open("setting.json", "r") as f:
            existing_settings = json.load(f)
    except FileNotFoundError:
        print("setting.json 未找到，將創建新的。")
    except json.JSONDecodeError:
        print("setting.json 格式錯誤，將使用新的設定覆蓋。")

    existing_settings.update(progress) # 用新的進度更新（或添加）到已有的設置中

    with open("setting.json", "w") as f:
        json.dump(existing_settings, f, indent=4) # 寫回更新後的完整設置

    notify_discord_webhook("中斷訊號捕捉到，已儲存模型與訓練進度。再見 👋")
    sys.exit(0)

signal.signal(signal.SIGINT, save_on_exit)
signal.signal(signal.SIGTERM, save_on_exit)

# --- 創建環境 ---
env = CryptoTradingEnv("technical/BTCUSDT_1d_technical.csv",data_split="train")
test_env = CryptoTradingEnv("technical/BTCUSDT_1d_technical.csv",data_split="test")

# --- 擬合或加載 Scaler ---
scaler_loaded_successfully = False # 標誌位，判斷是否成功加載

# 步驟 1: 嘗試加載
if os.path.exists(SCALER_PATH): # 先檢查文件是否存在
    try:
        loaded_scaler = joblib.load(SCALER_PATH)
        print(f"Scaler 加載自: {SCALER_PATH}")

        # 步驟 2: 如果加載成功，應用到環境
        env.scaler = loaded_scaler
        test_env.scaler = loaded_scaler # 應用同一個 scaler
        env._scaler_fitted = True
        test_env._scaler_fitted = True
        scaler_loaded_successfully = True
        print("Scaler 已成功加載並應用於環境。")

    except Exception as e:
        print(f"加載 Scaler 失敗 ({SCALER_PATH}): {e}. 將重新收集數據並擬合。")
        scaler_loaded_successfully = False # 確保標誌位為 False
else:
    print(f"Scaler 文件未找到: {SCALER_PATH}. 將收集數據並擬合。")
    scaler_loaded_successfully = False

# 步驟 3: 如果加載失敗，則執行數據收集、擬合和保存
if not scaler_loaded_successfully:
    print("開始收集狀態數據用於擬合 Scaler...")
    initial_states_for_scaler = []
    temp_env_for_scaler = None # 初始化，以便在 finally 中安全使用
    scaler_data = None       # 初始化

    try: # 將收集和擬合過程放在 try 塊中，以便使用 finally 清理
        temp_env_for_scaler = CryptoTradingEnv("technical/BTCUSDT_1d_technical.csv", data_split="train")
        temp_state_unscaled = temp_env_for_scaler.get_current_unscaled_state_vector()
        initial_states_for_scaler.append(temp_state_unscaled)

        done = False
        step_count = 0
        max_collect_steps = len(temp_env_for_scaler.data) - 1
        print(f"將在最多 {max_collect_steps} 步內收集狀態...")

        while not done and step_count < max_collect_steps:
            action = temp_env_for_scaler.action_space.sample()
            _, _, done, _ = temp_env_for_scaler.step(action)
            next_state_unscaled = temp_env_for_scaler.get_current_unscaled_state_vector()
            initial_states_for_scaler.append(next_state_unscaled)
            step_count += 1
            # if step_count % 500 == 0: # 可以取消註釋來查看進度
            #     print(f"已收集 {step_count} 個狀態...")

        print(f"總共收集了 {len(initial_states_for_scaler)} 個狀態用於擬合 Scaler。")
        scaler_data = np.array(initial_states_for_scaler)

        # 步驟 4: 擬合 Scaler
        if scaler_data.shape[0] > 0 and scaler_data.shape[1] == env.observation_space.shape[0]:
            print("正在擬合 Scaler...")
            env.fit_scaler(scaler_data) # <<< 擬合 env 的 scaler

            # 步驟 5: 應用到測試環境
            print("將擬合好的 Scaler 應用到測試環境...")
            test_env.scaler = env.scaler # <<< 將擬合好的 env.scaler 複製給 test_env
            test_env._scaler_fitted = True
            print("Scaler 擬合完成並已應用於訓練和測試環境。")

            # 步驟 6: 保存擬合好的 Scaler
            # 確保模型目錄存在
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
                print(f"創建目錄: {MODEL_PATH}")
            try:
                joblib.dump(env.scaler, SCALER_PATH) # <<< 保存擬合好的 env.scaler
                print(f"Scaler 已保存到: {SCALER_PATH}")
            except Exception as e:
                print(f"[錯誤] 保存 Scaler 失敗: {e}")

        else:
            # 如果收集數據失敗或維度不對
            print(f"[錯誤] 收集到的數據維度不正確或沒有收集到數據！Shape: {scaler_data.shape if scaler_data is not None else 'None'}, Expected columns: {env.observation_space.shape[0]}")
            exit() # 退出程序，因為無法進行歸一化

    finally: # 無論 try 中是否出錯，都執行清理
        # 步驟 7: 清理臨時數據
        print("清理臨時數據...")
        if temp_env_for_scaler is not None:
            del temp_env_for_scaler
        # 使用 'in locals()' 檢查變量是否存在，避免在異常情況下出錯
        if 'initial_states_for_scaler' in locals():
            del initial_states_for_scaler
        if 'scaler_data' in locals() and scaler_data is not None:
            del scaler_data
        gc.collect()
        print("臨時數據已清理。")


# --- 獲取環境信息 ---
state_shape = env.observation_space.shape
if len(state_shape) != 1:
     raise ValueError(f"預期狀態形狀為 1D (num_features,)，但得到的結果為 {state_shape}")
num_features = state_shape[0]
action_size = env.action_space.n
print(f"單步驟形狀：{state_shape}，特徵數量：{num_features}，動作數量：{action_size}")
print(f"使用序列長度：{SEQUENCE_LENGTH}")

# --- 繪製函數 (重用繪圖邏輯) ---
def plot_segment(plot_part_num, start_idx, end_idx, title_suffix,close_prices_all,mode="train"):
    plt.figure(figsize=(15, 7)) # 每個部分都創建一個新圖形
    plot_indices_segment = date[mode][start_idx:end_idx]#np.arange(start_idx, end_idx)
    close_prices_segment = close_prices_all[start_idx:end_idx]

    plt.plot(plot_indices_segment, close_prices_segment, label=f'{PRICE_COLUMN} Price', color='blue', alpha=0.8)

    # 過濾買賣點到當前段 (buy_positions 和 sell_positions 存儲的是索引)
    buy_indices_in_segment = [idx_tuple[1] for idx_tuple in buy_positions if start_idx <= idx_tuple[1] < end_idx]
    sell_indices_in_segment = [idx_tuple[1] for idx_tuple in sell_positions if start_idx <= idx_tuple[1] < end_idx]
    
    if len(buy_indices_in_segment) > 0:
        buy_prices_in_segment = close_prices_all[buy_indices_in_segment]
        # 買入點的 x 座標需要從 date 字典中根據索引獲取
        buy_x_coords = [date[mode][i] for i in buy_indices_in_segment] # <--- 修正這裡
        plt.scatter(buy_x_coords, buy_prices_in_segment, color='lime', marker='^', label='Buy', s=100, edgecolors='black')

    if len(sell_indices_in_segment) > 0:
        sell_prices_in_segment = close_prices_all[sell_indices_in_segment]
        # 賣出點的 x 座標需要從 date 字典中根據索引獲取
        sell_x_coords = [date[mode][i] for i in sell_indices_in_segment] # <--- 修正這裡
        plt.scatter(sell_x_coords, sell_prices_in_segment, color='red', marker='v', label='Sell', s=100, edgecolors='black')

    # 繪製買入標記並顯示買入比例
    for (action_ratio, idx) in buy_positions:
        if start_idx <= idx < end_idx:
            # 文本標記的 x 座標也需要從 date 字典中獲取
            text_x_coord = date[mode][idx] # <--- 修正這裡
            plt.text(text_x_coord, close_prices_all[idx] * 1.01, f"{action_ratio}", fontsize=9, color="green", ha="center")
    # 繪製賣出標記並顯示賣出比例
    for (action_ratio, idx) in sell_positions:
        if start_idx <= idx < end_idx:
            # 文本標記的 x 座標也需要從 date 字典中獲取
            text_x_coord = date[mode][idx] # <--- 修正這裡
            plt.text(text_x_coord, close_prices_all[idx] * 0.99, f"{action_ratio}", fontsize=9, color="red", ha="center")

    plt.title(f'Episode {episode+1} - Trading Actions {title_suffix}')
    plt.xlabel('Date')
    # 自動格式化日期標籤，使其更美觀
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 例如 YYYY-MM-DD
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator()) # 自動選擇合適的日期間隔
    plt.xticks(rotation=45) # 旋轉日期標籤以防重疊
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 儲存圖形，文件名包含部分編號
    # --- 儲存目錄 ---
    save_dir = f'.//training_plots//episode_{episode+1}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_filename_segment = os.path.join(save_dir, f"{mode}_episode_{episode+1}_part{plot_part_num}.png")
    plt.savefig(plot_filename_segment)
    plt.close() # 關閉當前圖形，釋放內存
    notify_discord_webhook(plot_filename_segment,"image")

# --- 清除終端機資訊 ---
def clear_lines(n):
    for _ in range(n):
        print("\033[F\033[K", end="")  # 上移一行 + 清除該行

# --- 價格數據用於繪圖 ---
PRICE_COLUMN = 'close'
if not hasattr(env, 'df') or not isinstance(env.df, pd.DataFrame):
     raise AttributeError("環境物件“env”必須具有一個屬性“df”，它是一個 Pandas DataFrame。")
if PRICE_COLUMN not in env.df.columns:
     raise ValueError(f"env.df 中未找到價格列“{PRICE_COLUMN}”")
close_prices_train = env.df[PRICE_COLUMN].values # 獲取所有收盤價用於繪圖
close_prices_test = test_env.df[PRICE_COLUMN].values # 獲取所有收盤價用於繪圖
date = {"train" : pd.to_datetime(env.df["timestamp"].values).to_list(), "test" : pd.to_datetime(test_env.df["timestamp"].values).to_list()}

# --- 初始化 LINE Bot v3 API ---
# 創建配置對象，並設置你的 Channel Access Token
configuration = Configuration(access_token=os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))

# 建立discord webhook作為line bot的備選方案
def notify_discord_webhook(msg, send_type="text"):
    try:
        url = os.getenv("DISCORD_WEBHOOK")
        if send_type == "text":
            headers = {"Content-Type": "application/json"}
            data = {"content": msg, "username": "訓練回報"}
            res = requests.post(url, headers = headers, json = data) 
        elif send_type == "image":
            #headers = {"Content-Type": "multipart/form-data"}
            with open(msg, 'rb') as f:
                files = {
                    'file': (msg, f)
                }
                data = {"content": "", "username": "訓練回報"}
                res = requests.post(url, data = data, files = files) 
    except:
        pass

# 創建 API 客戶端實例
api_client = ApiClient(configuration)

# 創建 MessagingApi 實例，傳入 API 客戶端
line_bot_api_v3 = MessagingApi(api_client) # 可以取一個新名字區分

# --- 使用 v3 API 推送消息 ---
try:
    if USE_LINE_BOT:
        # 創建 PushMessageRequest 對象
        push_message_request = PushMessageRequest(
            to=os.getenv('USER_ID'), # 指定接收者 USER_ID
            messages=[TextMessage(text="開始訓練!!!")] # 創建 TextMessage 對象列表 (即使只有一條消息也要是列表)
        )
        # 調用新的 API 實例的 push_message 方法
        line_bot_api_v3.push_message(push_message_request)
    else:
        print(f"use discord")
        notify_discord_webhook("開始訓練!!!")
except Exception as e:
    print(f"Error sending LINE notification change to discord") # 捕獲並打印可能的錯誤
    notify_discord_webhook("開始訓練!!!")

# --- 模型輸入形狀 ---
MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, num_features)

# --- 加載或創建模型 ---
try:
    model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
    # ... compile model ...
    print("繼續訓練模型!")
    # 如果是繼續訓練，最好也加載 target_model 的權重（如果之前有保存）
    try:
        target_model = tf.keras.models.load_model(MODEL_PATH + "_target", compile=False) # 假設保存為 _target 後綴
        print("目標網絡已加載！")
    except:
        print("無法加載目標網絡，將從主網絡複製權重。")
        target_model = build_inception_gru_dqn_model(input_shape=MODEL_INPUT_SHAPE, action_size=action_size)
        target_model.set_weights(model.get_weights()) # 初始化時權重一致
except Exception as e:
    print(f"無法加載模型 ({e}), 開始訓練新模型!")
    model = build_inception_gru_dqn_model(input_shape=MODEL_INPUT_SHAPE, action_size=action_size)
    # ... compile model ...
    target_model = build_inception_gru_dqn_model(input_shape=MODEL_INPUT_SHAPE, action_size=action_size)
    target_model.set_weights(model.get_weights()) # 初始化時權重一致
finally:
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR) # 使用初始學習率
    model.compile(optimizer=optimizer, loss='mse')

# 創建一個編譯後的預測函數
# --- 編譯後的預測函數簽名 ---
@tf.function(input_signature=[tf.TensorSpec(shape=(1, SEQUENCE_LENGTH, num_features), dtype=tf.float32)])
def compiled_predict(state_input):
  q_values = model(state_input, training=False)
  return q_values

@tf.function(input_signature=[tf.TensorSpec(shape=(None, SEQUENCE_LENGTH, num_features), dtype=tf.float32)])
def compiled_batch_predict(state_batch_input):
    """使用 tf.function 編譯模型的批次預測"""
    # 直接調用模型，設置 training=False
    q_values = model(state_batch_input, training=False)
    return q_values

@tf.function(input_signature=[tf.TensorSpec(shape=(None, SEQUENCE_LENGTH, num_features), dtype=tf.float32)])
def compiled_batch_predict_target(state_batch_input):
    """使用 tf.function 編譯目標網絡的批次預測"""
    q_values = target_model(state_batch_input, training=False) # 使用 target_model
    return q_values

# --- 新增：評估函數 ---
def evaluate_model(trained_model, eval_env, sequence_length, num_features, eval_episodes=1):
    """在測試環境上評估訓練好的模型"""
    print(f"\n--- 開始對測試集進行評估 ({eval_episodes} episodes)) ---")
    total_rewards = []
    final_balances = []
    max_drawdowns = []
    peak_balances = []
    all_hight_sells = []
    all_low_sells = []
    all_steps = []

    # 創建評估用的預測函數 (使用傳入的 trained_model)
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, sequence_length, num_features), dtype=tf.float32)])
    def eval_predict(state_input):
        return trained_model(state_input, training=False)

    for i in range(eval_episodes):
        eval_raw_state = eval_env.reset()
        eval_state_deque = deque(maxlen=sequence_length)
        for _ in range(sequence_length):
            eval_state_deque.append(eval_raw_state)
        eval_state_sequence = np.array(eval_state_deque)

        eval_done = False
        eval_total_reward = 0
        eval_t = 0
        global buy_positions # <<< 初始化 episode 的買入點記錄
        buy_positions = []
        global sell_positions # <<< 初始化 episode 的賣出點記錄
        sell_positions = []
        
        while not eval_done:
            eval_t += 1
            eval_model_input = eval_state_sequence.reshape(1, sequence_length, num_features).astype(np.float32)
            # *** 評估時關閉探索 (Epsilon=0) ***
            eval_q_values_tensor = eval_predict(tf.convert_to_tensor(eval_model_input, dtype=tf.float32))
            eval_action = np.argmax(eval_q_values_tensor.numpy()[0])

            # *** 注意：評估時不應有動作修正，以反映模型真實決策 ***
            #if eval_env.balance <= 0 and (0 < eval_action < 6): eval_action = 0
            #elif eval_env.position <= 0 and (5 < eval_action < 11): eval_action = 0

            eval_next_raw_state, eval_reward, eval_done, eval_trade_executed = eval_env.step(eval_action)

            if eval_trade_executed:
                if 0 < eval_action < 6:
                    current_data_index = eval_env.current_step
                    buy_positions.append((eval_env.action_map[eval_action], current_data_index)) # 使用 eval_action
                elif 5 < eval_action < 11 :
                    current_data_index = eval_env.current_step
                    sell_positions.append((eval_env.action_map[eval_action], current_data_index)) # 使用 eval_action

            eval_state_deque.append(eval_next_raw_state)
            eval_next_state_sequence = np.array(eval_state_deque)

            eval_total_reward += eval_reward
            eval_state_sequence = eval_next_state_sequence

        data_len = len(close_prices_test)
        # 如果數據不足 TIME_RANGE，只繪製可用的數據
        actual_range = data_len
        steps_per_plot = actual_range // 4
        remaining_steps = actual_range % 4 # 處理餘數情況

        # --- 計算並繪製第一部分 ---
        plot1_start_index = 0
        # 第一部分包含前半部分 + 可能多餘的一步 (如果 TIME_RANGE 是奇數)
        plot1_end_index = steps_per_plot + remaining_steps
        if plot1_end_index > plot1_start_index: # 確保範圍有效
            plot_segment(1, plot1_start_index, plot1_end_index, f"(Steps {plot1_start_index}-{plot1_end_index-1})",close_prices_test,"test")
        # --- 計算並繪製第二部分 ---
            plot2_start_index = plot1_end_index
        for i in range(actual_range // steps_per_plot - 2):
                plot2_end_index = plot1_end_index + (i+2) * steps_per_plot # 確保結束點正確
                if plot2_end_index > plot2_start_index: # 確保範圍有效
                    plot_segment(i+2, plot2_start_index, plot2_end_index, f"(Steps {plot2_start_index}-{plot2_end_index-1})",close_prices_test,"test")
                plot2_start_index = plot2_end_index

        # 記錄評估結果
        total_rewards.append(eval_total_reward)
        final_balances.append(eval_env.total_balance)
        max_drawdowns.append(eval_env.max_drawdown) # 環境在 done 時計算
        peak_balances.append(np.max(eval_env.balance_history)) # 從歷史記錄計算峰值
        all_hight_sells.append(eval_env.hight_sell_timer)
        all_low_sells.append(eval_env.low_sell_timer)
        all_steps.append(eval_t)
        print(f"Eval Ep {i+1}: Steps={eval_t}, Reward={eval_total_reward:.2f}, Balance={eval_env.total_balance:.2f}, MaxDrawdown={eval_env.max_drawdown*100:.2f}%, Peak={np.max(eval_env.balance_history):.2f}")

    # 計算平均結果
    avg_reward = np.mean(total_rewards)
    avg_balance = np.mean(final_balances)
    avg_max_drawdown = np.mean(max_drawdowns)
    avg_peak = np.mean(peak_balances)
    print(f"--- Evaluation Summary (Avg over {eval_episodes} episodes) ---")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Final Balance: {avg_balance:.2f}")
    print(f"Avg Max Drawdown: {avg_max_drawdown*100:.2f}%")
    print(f"Avg Peak Balance: {avg_peak:.2f}")
    print(f"----------------------------------------------------")
    return {"avg_reward": avg_reward, "avg_balance": avg_balance, "avg_max_drawdown": avg_max_drawdown}

# --- 經驗回放池 ------
best_balance_memory = deque(maxlen=MEMORY_SIZE)
best_reward_memory = deque(maxlen=MEMORY_SIZE)
memory = deque(maxlen=MEMORY_SIZE)

# --- 獲取當前進程對象 ---
current_process = psutil.Process(os.getpid())

# --- 訓練過程 ---
global_step_count = setting.get("global_step_count", 0) # <--- 需要一個全局步數計數器
current_lr = 0

for episode in range(START_EPISODE,EPISODES):
    # --- 狀態初始化 ---
    raw_state = env.reset() # 獲取第一個原始狀態 (num_features,)
    # 獲取當前進程使用的物理內存 (RSS - Resident Set Size)
    mem_info = current_process.memory_info()
    rss_gb = mem_info.rss / (1024**3) # 轉換為 GB
    # 使用一個 deque 來高效地維護狀態序列
    state_sequence_deque = deque(maxlen=SEQUENCE_LENGTH)
    # 用初始狀態填充隊列 (重複第一個狀態 SEQUENCE_LENGTH 次)
    for _ in range(SEQUENCE_LENGTH):
        state_sequence_deque.append(raw_state)
    # 將 deque 轉換為 NumPy 數組，作為初始的狀態序列
    state_sequence = np.array(state_sequence_deque) # Shape: (SEQUENCE_LENGTH, num_features)

    # --- 計算當前回合的 Epsilon (基於回合數的週期性退火) ---
    progress_in_cycle = (episode % EPSILON_CYCLE_LENGTH) / EPSILON_CYCLE_LENGTH
    cosine_decay_epsilon = 0.5 * (1 + np.cos(np.pi * progress_in_cycle))
    current_epsilon_dynamic = (INITIAL_EPSILON - MIN_EPSILON_CYCLE) * cosine_decay_epsilon + MIN_EPSILON_CYCLE
    EPSILON = max(current_epsilon_dynamic, EPSILON_MIN) # 確保不低於全局最小值

    total_reward = 0
    buy_positions = [] # <<< 初始化 episode 的買入點記錄
    sell_positions = [] # <<< 初始化 episode 的賣出點記錄
    current_episode_experiences = [] # <--- 暫存當前回合經驗
    done = False
    t = 0 # 時間步計數器

    # --- 使用標準 RL 循環 ---
    while not done:
        t += 1
        global_step_count += 1 # 更新全局步數
        # --- 修改：準備模型輸入 ---
        # 將 state_sequence reshape 成模型期望的輸入 (1, sequence_length, num_features)
        model_input_state = state_sequence.reshape(1, SEQUENCE_LENGTH, num_features).astype(np.float32)
        # 探索或利用策略
        if np.random.rand() < EPSILON:
            action = np.random.choice(action_size)
        else:
            # 使用編譯後的函數進行預測
            q_values_tensor = compiled_predict(tf.convert_to_tensor(model_input_state, dtype=tf.float32))
            action = np.argmax(q_values_tensor.numpy()[0])

        if env.balance <= 0 and (0 < action < 6):
            action = 0
        elif env.position <= 0 and (5 < action < 11):
            action = 0

        # 執行選擇的動作 (可能是修改後的 action)
        next_state_raw, reward, done, trade_executed = env.step(action)

        # --- 修改：更新狀態序列 ---
        # 將新的原始狀態添加到 deque 的末尾 (最舊的會自動移除)
        state_sequence_deque.append(next_state_raw)
        # 將更新後的 deque 轉換為 NumPy 數組，作為 next_state_sequence
        next_state_sequence = np.array(state_sequence_deque) # Shape: (SEQUENCE_LENGTH, num_features)
        
        # --- 記錄交易行為 ---
        # 使用執行的動作 action 來更新狀態和記錄位置
        if trade_executed:
            if 0 < action < 6:  # 實際執行了買入
                # 使用 env 內部的時間索引或者 t-1 (因為 t 是當前步數，動作發生在 t-1 結束時)
                # 假設 env.current_step 記錄了 DataFrame 的索引
                # 如果沒有，用 t-1 可能會有偏差，取決於 env 如何實現
                current_data_index = env.current_step # 假設環境有這個屬性
                buy_positions.append((env.action_map[action],current_data_index)) # 記錄數據的索引
            elif 5 < action < 11 :  # 實際執行了賣出
                current_data_index = env.current_step
                sell_positions.append((env.action_map[action],current_data_index))

        # 存儲經驗 (原始 state, 實際執行的 action, reward, next_state, done)
        memory.append((state_sequence, action, reward, next_state_sequence, done))
        current_episode_experiences.append((state_sequence, action, reward, next_state_sequence, done))

        total_reward += reward
        state_sequence = next_state_sequence # *** 更新 state_sequence 以進行下一步 ***

        # --- 在每一步後嘗試訓練 ---
        if len(memory) >= BATCH_SIZE and (t-1)%7 == 0: # 確保三個池子都有數據
            # --- 混合抽樣 ---
            can_mixed_sample = len(memory) >= BATCH_SIZE // 2 and (len(best_balance_memory) > 0 or len(best_reward_memory) > 0)

            if can_mixed_sample:
                # 1. 計算各來源的目標樣本數量 (確保為整數且總和為 BATCH_SIZE)
                main_target_size = int(BATCH_SIZE * 0.9)
                remaining_size = BATCH_SIZE - main_target_size

                # 按比例分配剩餘名額，處理取整
                balance_target_size = int(remaining_size * 0.8)
                # 將餘數分配給 reward 池，確保總數正確
                reward_target_size = remaining_size - balance_target_size
            else:
                main_target_size = BATCH_SIZE

            # (可選) 檢查主池實際大小，如果不足 main_target_size，需要調整
            if len(memory) < main_target_size:
                print(f"[警告] 主經驗池樣本不足 {main_target_size}，實際只有 {len(memory)}")
                main_target_size = len(memory) # 只能抽取這麼多
                # 重新計算剩餘的，並重新分配給 best pools
                remaining_size = BATCH_SIZE - main_target_size
                balance_target_size = int(remaining_size * 0.8)
                reward_target_size = remaining_size - balance_target_size

            # 2. 從主經驗池抽樣
            main_minibatch = random.sample(memory, main_target_size)

            # 3. 從 Best Balance 池抽樣
            if len(best_balance_memory) > 0:
                # 無論數量是否足夠，都使用 random.choices (允許重複) 抽取目標數量
                balance_batch = random.choices(list(best_balance_memory), k=balance_target_size)
            else:
                balance_batch = [] # 如果池子是空的

            # 4. 從 Best Reward 池抽樣
            if len(best_reward_memory) > 0:
                reward_batch = random.choices(list(best_reward_memory), k=reward_target_size)
            else:
                reward_batch = [] # 如果池子是空的

            # 5. 合併初步的 minibatch
            minibatch = main_minibatch + balance_batch + reward_batch

            # 6. 處理因最佳池為空導致的樣本不足情況
            current_size = len(minibatch)
            missing_count = BATCH_SIZE - current_size

            if missing_count > 0:
                print(f"混合抽樣後樣本不足，需要從主池補充 {missing_count} 個")
                # 從主池補充缺少的樣本
                if len(memory) >= main_target_size + missing_count: # 檢查主池是否有足夠的額外樣本
                    # 嘗試抽取與 main_minibatch 不重複的樣本 (較複雜)
                    # 簡化處理：直接再抽 missing_count 個，允許少量重複
                    supplement_batch = random.sample(memory, missing_count)
                    minibatch.extend(supplement_batch)
                elif len(memory) > main_target_size: # 如果只能補充部分
                    available_supplement = len(memory) - main_target_size
                    supplement_batch = random.sample(memory, available_supplement)
                    minibatch.extend(supplement_batch)
                    print(f"[警告] 主池只能補充 {available_supplement} 個，最終批次大小為 {len(minibatch)}")
            
            # 7. 打亂最終的 minibatch 順序
            random.shuffle(minibatch)
            actual_batch_size = len(minibatch)

            # --- 計算當前學習率 ---
            # 使用 global_step_count 作為 cosine_decay 的輸入
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (global_step_count % EPOCHS_DROP_RL) / EPOCHS_DROP_RL))
            # 讓學習率從 INITIAL_LR 降到 ALPHA_LR
            # new_lr = (INITIAL_LR - ALPHA_LR) * cosine_decay + ALPHA_LR
            # 或者使用你的 drop_rate 模式，但終點設為 ALPHA_LR
            decayed = (1 - LR_DROP_RATE) * cosine_decay + LR_DROP_RATE # 這裡 drop_rate 代表最低比例
            new_lr = INITIAL_LR * decayed
            current_lr = max(new_lr, ALPHA_LR) # 確保不低於最小學習率

            # --- 更新優化器的學習率 ---
            tf.keras.backend.set_value(model.optimizer.learning_rate, current_lr)

            # --- 批次處理 ---
            # 0. 獲取實際的批次大小
            actual_batch_size = len(minibatch)

            # 1. 從 minibatch 中解包數據
            state_sequences_batch = np.array([transition[0] for transition in minibatch])
            actions_batch = np.array([transition[1] for transition in minibatch])
            rewards_batch = np.array([transition[2] for transition in minibatch])
            next_state_sequences_batch = np.array([transition[3] for transition in minibatch])
            dones_batch = np.array([transition[4] for transition in minibatch])

            # --- DDQN 核心計算 ---
            # 2. 使用主網絡 (model) 預測當前狀態 (s) 的 Q 值
            #    Q_online(s, a)
            q_values_current_tensor = compiled_batch_predict(
                tf.convert_to_tensor(state_sequences_batch, dtype=tf.float32)
            )
            q_values_current = q_values_current_tensor.numpy() # 轉換為 NumPy 數組

            # 3. 使用主網絡 (model) 預測下一個狀態 (s') 的 Q 值，用於選擇動作 a_max
            #    Q_online(s', a)
            q_values_next_online_tensor = compiled_batch_predict(
                tf.convert_to_tensor(next_state_sequences_batch, dtype=tf.float32)
            )
            q_values_next_online = q_values_next_online_tensor.numpy() # 轉換為 NumPy 數組

            # 4. 使用目標網絡 (target_model) 預測下一個狀態 (s') 的 Q 值，用於評估選定動作的價值
            #    Q_target(s', a)
            q_values_next_target_tensor = compiled_batch_predict_target( # 使用目標網絡的預測函數
                tf.convert_to_tensor(next_state_sequences_batch, dtype=tf.float32)
            )
            q_values_next_target = q_values_next_target_tensor.numpy() # 轉換為 NumPy 數組
            # --- DDQN 核心計算結束 ---

            # 5. 初始化 targets 數組，其基礎是當前狀態的 Q 值
            targets = np.copy(q_values_current)

            # 6. 計算目標 Q 值 (TD Target) - 使用 DDQN 邏輯
            #    循環使用實際的批次大小
            for i in range(actual_batch_size):
                if dones_batch[i]:
                    # 如果是終止狀態，目標 Q 值就是即時獎勵
                    targets[i, actions_batch[i]] = rewards_batch[i]
                else:
                    # DDQN 步驟：
                    # a_max = argmax_a' Q_online(s'[i], a')
                    # 從主網絡對下一個狀態的 Q 值預測中，找到使 Q 值最大的動作的索引
                    action_max_online = np.argmax(q_values_next_online[i])

                    # Target_DDQN = R[i] + γ * Q_target(s'[i], a_max)
                    # 使用目標網絡對下一個狀態的 Q 值預測，來獲取 action_max_online 對應的 Q 值
                    targets[i, actions_batch[i]] = rewards_batch[i] + GAMMA * q_values_next_target[i, action_max_online]

            # 7. 批次訓練主網絡 (model)
            #    使用實際的批次大小
            model.fit(state_sequences_batch, targets, epochs=1, verbose=0, batch_size=actual_batch_size)

            if t > 1:
                clear_lines(8)  # 清除上次輸出的 10 行
            #顯示目前回合的即時訓練成績
            print(f"🚅{datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"回合{episode+1}/{EPISODES}，探索機率：{EPSILON:.4f}，學習率: {current_lr:.6f}")
            print(f"步數：{t+1}/{BEST_STEPS}")
            print(f"總獎勵：{total_reward:.2f}/{BEST_REWARD:6.2f}")
            print(f"交易分數：{env.cumulative_trade_reward:6.2f}，資產變化分數：{env.cumulative_asset_change_reward:6.2f}，持有與不動作分數：{env.cumulative_hold_penalty:6.2f}，回徹逞罰分數：{env.cumulative_drawdown_penalty:6.2f}，無效交易懲罰：{env.cumulative_invalid_trade_penalty:6.2f}")
            print(f"高賣次數：{env.hight_sell_timer}，低賣次數：{env.low_sell_timer}，比例{(env.hight_sell_timer/max(env.low_sell_timer,1)):6.2f}")
            print(f"RAM：{rss_gb:.2f}/{MEMORY_LIMIT_GB}")
            print(f"餘額：{env.total_balance:6.2f}/{BEST_BALANCE:6.2f}，餘額歷史最高值：{np.max(env.balance_history):6.2f}，最大回撤：{env.max_drawdown:6.2f}")

    target_model.set_weights(model.get_weights())

    #顯示與傳送每回合的訓練成績
    clear_lines(8)
    report_text = f"""🚩{datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}
    回合{episode+1}/{EPISODES}，探索機率：{EPSILON:.4f}，學習率: {current_lr:.6f}
    步數：{t}/{BEST_STEPS}
    總獎勵：{total_reward:.2f}/{BEST_REWARD:6.2f}
    交易分數：{env.cumulative_trade_reward:6.2f}，資產變化分數：{env.cumulative_asset_change_reward:6.2f}，持有與不動作分數：{env.cumulative_hold_penalty:6.2f}，回徹逞罰分數：{env.cumulative_drawdown_penalty:6.2f}，無效交易懲罰：{env.cumulative_invalid_trade_penalty:6.2f}
    高賣次數：{env.hight_sell_timer}，低賣次數：{env.low_sell_timer}，比例{(env.hight_sell_timer/max(env.low_sell_timer,1)):6.2f}
    RAM：{rss_gb:.2f}/{MEMORY_LIMIT_GB}
    餘額：{env.total_balance:6.2f}/{BEST_BALANCE:6.2f}，餘額歷史最高值：{np.max(env.balance_history):6.2f}，最大回撤：{env.max_drawdown:6.2f}
    🔚
    """
    print(report_text)

    # --- Episode 結束 ---

    # --- 使用 v3 API 推送消息 ---
    try:
        if USE_LINE_BOT:
            # 創建 PushMessageRequest 對象
            push_message_request = PushMessageRequest(
                to=os.getenv('USER_ID'), # 指定接收者 USER_ID
                messages=[TextMessage(text=report_text)] # 創建 TextMessage 對象列表 (即使只有一條消息也要是列表)
            )
            # 調用新的 API 實例的 push_message 方法
            line_bot_api_v3.push_message(push_message_request)
        else:
            notify_discord_webhook(report_text)

    except:
         notify_discord_webhook(report_text)
    
    # --- 繪圖 (每 25 episodes) ---
    try:
        if episode % 25 == 0 or t > BEST_STEPS or total_reward > BEST_REWARD or env.total_balance > BEST_BALANCE:
            if BEST_STEPS > TIME_RANGE:
                TIME_RANGE = BEST_STEPS
                NUM_PLOTS = TIME_RANGE//100 
            data_len = len(close_prices_train)
            # 如果數據不足 TIME_RANGE，只繪製可用的數據
            actual_range = min(TIME_RANGE, data_len)
            steps_per_plot = actual_range // NUM_PLOTS
            remaining_steps = actual_range % NUM_PLOTS # 處理奇數情況

            # --- 計算並繪製第一部分 ---
            plot1_start_index = 0
            # 第一部分包含前半部分 + 可能多餘的一步 (如果 TIME_RANGE 是奇數)
            plot1_end_index = steps_per_plot + remaining_steps #250
            if plot1_end_index > plot1_start_index: # 確保範圍有效
                plot_segment(1, plot1_start_index, plot1_end_index, f"(Steps {plot1_start_index}-{plot1_end_index-1})",close_prices_train)
            # --- 計算並繪製第二部分 ---
                plot2_start_index = plot1_end_index
            for i in range(NUM_PLOTS - 2):
                    plot2_end_index = plot1_end_index + (i+2) * steps_per_plot # 確保結束點正確
                    if plot2_end_index > plot2_start_index: # 確保範圍有效
                        plot_segment(i+2, plot2_start_index, plot2_end_index, f"(Steps {plot2_start_index}-{plot2_end_index-1})",close_prices_train)
                    plot2_start_index = plot2_end_index
    except Exception as e:
        print(f"繪圖發生錯誤:{e}")


    # --- 定期保存模型 ---
    if episode % 50 == 0 or t > BEST_STEPS or total_reward > BEST_REWARD or env.total_balance > BEST_BALANCE:
        print(f"Saving model at episode {episode+1}...")
        if t > BEST_STEPS:
            BEST_STEPS = t
            model.save((MODEL_PATH + "_BEST_STEP"), save_format='tf')
            notify_discord_webhook("GET NEW BEST STEP 🎉")
        if total_reward > BEST_REWARD:
            BEST_REWARD = total_reward
            #best_reward_memory.clear() # 清空舊的最佳回合經驗
            for exp in current_episode_experiences: # 將新最佳回合經驗存入
                best_reward_memory.append(exp)
            model.save((MODEL_PATH + "_BEST_REWARD"), save_format='tf')
            notify_discord_webhook("GET NEW BEST REWARD 🎉")
        if env.total_balance > BEST_BALANCE:
            BEST_BALANCE = env.total_balance
            #best_balance_memory.clear() # 清空舊的最佳回合經驗
            for exp in current_episode_experiences: # 將新最佳回合經驗存入
                best_balance_memory.append(exp)
            model.save((MODEL_PATH + "_BEST_BALANCE"), save_format='tf')
            notify_discord_webhook("GET NEW BEST BALANCE 🎉")

        model.save(MODEL_PATH, save_format='tf')
        eval_results = evaluate_model(model, test_env, SEQUENCE_LENGTH, num_features)
        eval_report = f"""
        評估結果📈 - 
        平均獎勵：{eval_results['avg_reward']:.2f}，平均餘額：{eval_results['avg_balance']:.2f}，平均最大回撤：{eval_results['avg_max_drawdown']*100:.2f}%
        """
        notify_discord_webhook(eval_report)
        

    # --- 可選的記憶體清理 ---
    if rss_gb > MEMORY_LIMIT_GB:
        del minibatch, actions_batch, rewards_batch, dones_batch
        del q_values_current, targets
        try:
            current_model_path = MODEL_PATH # 或者一個臨時路徑
            model.save(current_model_path) # 保存當前模型狀態
            del model
            tf.keras.backend.clear_session()
            gc.collect()
            model = tf.keras.models.load_model(current_model_path, compile=False)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse') # <--- 需要重新編譯
            # 可能還需要重新創建 compiled_predict 函數，因為它依賴全局 model
            @tf.function(input_signature=[tf.TensorSpec(shape=(1, SEQUENCE_LENGTH, num_features), dtype=tf.float32)])
            def compiled_predict(state_input):
                q_values = model(state_input, training=False)
                return q_values

            @tf.function(input_signature=[tf.TensorSpec(shape=(None, SEQUENCE_LENGTH, num_features), dtype=tf.float32)])
            def compiled_batch_predict(state_batch_input):
                """使用 tf.function 編譯模型的批次預測"""
                # 直接調用模型，設置 training=False
                q_values = model(state_batch_input, training=False)
                return q_values
        except Exception as load_e:
            print(f"[Error] Failed to reload model after cleanup: {load_e}")
            # 這裡需要錯誤處理，是退出還是嘗試繼續？
            raise load_e # 或者 exit()
        
# --- 最終保存模型 ---
print("Training finished. Saving final model...")
model.save(MODEL_PATH)
print("Model saved.")