import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler # 用於歸一化

class CryptoTradingEnv(gym.Env):
    def __init__(self, data_file, data_split='train', train_ratio=0.8, initial_balance=500):
        super(CryptoTradingEnv, self).__init__()

        # --- 數據加載與劃分 (修正測試集邏輯) ---
        full_df = pd.read_csv(data_file)
        split_index = int(len(full_df) * train_ratio)
        if data_split == 'train':
            self.df = full_df[:split_index].reset_index(drop=True)
        elif data_split == 'test':
            self.df = full_df[split_index:].reset_index(drop=True) # <<< 修正
        else:
            self.df = full_df

        # --- 選擇特徵 ---
        self.feature_columns = [
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
        self.data = self.df[self.feature_columns].values
        self.num_data_features = self.data.shape[1] # 原始數據特徵數量

        # --- 初始化 scaler ---
        self.scaler = StandardScaler() # 狀態歸一化器

        # --- 擬合 Scaler (僅使用訓練數據！) ---
        # 這一步很重要，確保歸一化基於訓練數據的分佈
        # 你需要在創建 train_env 後，在外部調用 fit_scaler
        # 或者在 __init__ 中加載預先擬合好的 scaler 參數
        self._scaler_fitted = False # 標記 scaler 是否已擬合

        self.initial_balance = initial_balance
        # ... (action_map) ...
        self.action_map = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0, 6: -0.2, 7: -0.4, 8: -0.6, 9: -0.8, 10: -1.0}

        # --- 觀察空間 (維度 = 數據特徵 + 4 個額外狀態) ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_data_features + 4,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_map))

        # --- 狀態變數初始化 ---
        self._reset_state() # 將初始化邏輯放入一個輔助方法

    def get_current_unscaled_state_vector(self):
        """獲取當前未經縮放的、用於 Scaler 擬合的完整狀態向量"""
        # --- 獲取 price ---
        if self.current_step < len(self.data):
            current_data = self.data[self.current_step]
            price = (current_data[1] + current_data[3]) / 2
        else:
            price = 0 # 或者使用其他邊界值處理

        # --- 獲取原始數據特徵 ---
        if self.current_step >= len(self.data):
            raw_features = np.zeros(self.num_data_features)
        else:
            raw_features = self.data[self.current_step]

        # --- 計算額外狀態特徵 (與 _get_normalized_state 保持一致) ---
        current_total_balance = self.total_balance
        if current_total_balance > 1e-9:
            portfolio_crypto_pct = (self.position * price) / current_total_balance
            portfolio_crypto_pct = np.clip(portfolio_crypto_pct, 0.0, 1.0)
        else:
            portfolio_crypto_pct = 0.0

        if self.initial_balance > 1e-9:
            total_balance_ratio = current_total_balance / self.initial_balance
        else:
            total_balance_ratio = 1.0

        current_no_action_timer = self.no_action_timer

        # 4. 計算符號平方根相對成本
        current_buy_point = self.buy_point
        if current_buy_point > 1e-9 and price > 1e-9: # 確保持倉且價格有效
            relative_buy_point = (price - current_buy_point) / price
            # 計算符號平方根
            signed_sqrt_relative_buy_point = np.sign(relative_buy_point) * np.sqrt(np.abs(relative_buy_point))
        else: # 空倉或價格無效
            signed_sqrt_relative_buy_point = 0.0 # 空倉/無效時設為 0

        # --- 組合狀態向量 ---
        current_state_unscaled = np.concatenate([
            raw_features,
            [portfolio_crypto_pct, total_balance_ratio, current_no_action_timer,signed_sqrt_relative_buy_point]
        ]).astype(np.float32)
        return current_state_unscaled

    # --- 新增：擬合 Scaler 的方法 ---
    def fit_scaler(self, scaler_data):
        """使用提供的數據來擬合 Scaler"""
        # scaler_data 應該是包含所有狀態特徵 (包括 balance, position, total_balance) 的 N x D 數組
        # 通常使用訓練環境的數據來擬合
        if scaler_data.shape[1] != self.observation_space.shape[0]:
             raise ValueError("數據維度與觀察空間不匹配")
        self.scaler.fit(scaler_data)
        self._scaler_fitted = True
        print("Scaler fitted successfully.")

    def _get_normalized_state(self):
        """獲取當前狀態並進行歸一化"""
        # --- 獲取 price (只在需要時計算一次) ---
        # 邊界情況處理：如果 current_step 超出範圍，無法獲取當前 price
        # 可以使用最後一步的 price，或者根據情況設置 price 為 0 或其他值
        # 這裡我們假設在 done=True 之前，price 總能計算
        if self.current_step < len(self.data):
            current_data = self.data[self.current_step]
            price = (current_data[1] + current_data[3]) / 2 # 使用 high 和 close 計算價格
        else:
            # 在 episode 結束後，使用最後一次的 price 或其他默認值
            # 這裡我們假設用 0，因為無法交易了
            price = 0 # 或者 self.balance_history[-1] / (self.position + self.balance/price_last) ?
                    # 為了簡化，用 0 處理邊界狀態

        # --- 獲取原始數據特徵 ---
        if self.current_step >= len(self.data):
            raw_features = np.zeros(self.num_data_features)
        else:
            raw_features = self.data[self.current_step]

        # --- 計算額外狀態特徵 ---
        # 1. 持有幣價值佔比 (處理 total_balance 為 0)
        current_total_balance = self.total_balance # 確保使用更新後的 total_balance
        if current_total_balance > 1e-9: # 避免除以零
            portfolio_crypto_pct = (self.position * price) / current_total_balance
            # 確保比例在合理範圍 (例如，浮點數誤差可能導致略微超出 [0,1])
            portfolio_crypto_pct = np.clip(portfolio_crypto_pct, 0.0, 1.0)
        else:
            portfolio_crypto_pct = 0.0 # 如果總資產為 0，則持有比例為 0

        # 2. 總資金/起始資金比 (處理 initial_balance 為 0)
        if self.initial_balance > 1e-9:
            total_balance_ratio = current_total_balance / self.initial_balance
        else:
            total_balance_ratio = 1.0 # 如果初始資金為 0，比值設為 1

        # 3. 不活躍次數
        current_no_action_timer = self.no_action_timer

        # 4. 計算符號平方根相對成本
        current_buy_point = self.buy_point
        if current_buy_point > 1e-9 and price > 1e-9: # 確保持倉且價格有效
            relative_buy_point = (price - current_buy_point) / price
            # 計算符號平方根
            signed_sqrt_relative_buy_point = np.sign(relative_buy_point) * np.sqrt(np.abs(relative_buy_point))
        else: # 空倉或價格無效
            signed_sqrt_relative_buy_point = 0.0 # 空倉/無效時設為 0

        # --- 組合未縮放狀態向量 ---
        current_state_unscaled = np.concatenate([
            raw_features,
            [portfolio_crypto_pct, total_balance_ratio, current_no_action_timer,signed_sqrt_relative_buy_point]
        ]).astype(np.float32)

        # --- 歸一化 ---
        if not self._scaler_fitted:
            return current_state_unscaled
        else:
            scaled_state = self.scaler.transform(current_state_unscaled.reshape(1, -1))
            return scaled_state.flatten()

    def _reset_state(self):
        """重置所有內部狀態"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.buy_point = 0
        self.total_balance = self.initial_balance
        self.no_action_timer = 0
        self.hight_sell_timer = 0
        self.low_sell_timer = 0
        self.balance_history = [self.initial_balance] # 初始化資產歷史
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance # 初始化峰值
        self.cumulative_trade_reward = 0.0 #交易分數
        self.cumulative_asset_change_reward = 0.0 #資產變化分數
        self.cumulative_hold_penalty = 0.0 #持有與不動作分數
        self.cumulative_drawdown_penalty = 0.0 #回徹逞罰分數
        self.cumulative_invalid_trade_penalty = 0.0 # 無效交易懲罰

    def reset(self):
        self._reset_state()
        # 返回歸一化後的初始狀態
        return self._get_normalized_state()

    def step(self, action):
        # 記錄上一步狀態，用於獎勵計算和歷史記錄
        previous_total_balance = self.total_balance
        # --- 確保索引有效 ---
        if self.current_step >= len(self.data):
             # 如果步數已經超出範圍，這是異常情況，因為 done=True 應該在前一步返回
             # 可以返回一個表示錯誤或終止的狀態
             print(f"[Error] Step called at invalid current_step: {self.current_step}")
             final_state = self._get_normalized_state() # 獲取最後狀態的歸一化值
             return final_state, 0, True, {"error": "Invalid step"}

        current_data = self.data[self.current_step]
        price = (current_data[1] + current_data[3]) / 2 # 使用 high 和 close 計算價格
        self.no_action_timer += 1
        trade_ratio = self.action_map[action]
        reward = 0.0 # 初始化為浮點數
        invalid_trade_penalty = 0.0 # 新增局部變量

        # --- 交易邏輯 ---
        trade_executed = False # 標記是否執行了交易
        trade_reward = 0.0     # 初始化交易盈虧獎勵

        # 買入
        if trade_ratio > 0 and self.balance > 1e-6: # 增加浮點數判斷
            buy_amount = self.balance * trade_ratio # min 不需要
            # 考慮最小交易金額？ (可選)
            if buy_amount > price * 1e-6: # 假設最小交易額是0.000001btc
                buy_units = (buy_amount / price) * 0.999
                total_position_value = self.position * self.buy_point
                self.position += buy_units
                # 加權平均成本計算需要處理 position 為 0 的情況
                if self.position > 1e-9:
                    self.buy_point = (total_position_value + buy_amount * (1/0.999)) / self.position # 成本應基於未扣手續費的金額
                else:
                    self.buy_point = price / 0.999 # 首次買入成本近似
                self.balance -= buy_amount
                if self.balance < 1e-6: self.balance = 0
                self.no_action_timer = 0
                trade_executed = True
            else:
                trade_ratio = 0 # 買入金額太小，視為無效
                reward -= 0.01

        # 賣出
        elif trade_ratio < 0 and self.position > 1e-6: # 增加浮點數判斷
            sell_units = self.position * abs(trade_ratio) # min 不需要
            if sell_units > 1e-6: # 確保有足夠倉位賣
                sell_amount = (sell_units * price) * 0.999
                self.balance += sell_amount

                # 計算交易盈虧獎勵 (基於對數回報)
                if self.buy_point > 0:
                    profit_ratio = (price * 0.999) / self.buy_point
                    if profit_ratio > 0:
                        log_return = np.log(profit_ratio)
                        if self.no_action_timer < 15:
                            LOG_TRADE_REWARD_SCALE = min(125,self.no_action_timer * 8) # <<<=== **可調參數 1**
                        else:
                            LOG_TRADE_REWARD_SCALE = 125
                        LOSS_PENALTY_MULTIPLIER = 1.1 # <<<=== **可調參數 2**
                        if log_return > 0:
                            trade_reward = log_return * LOG_TRADE_REWARD_SCALE
                            self.hight_sell_timer += 1
                        else:
                            trade_reward = log_return * LOG_TRADE_REWARD_SCALE * LOSS_PENALTY_MULTIPLIER # log_return 是負數，乘以正數仍是負獎勵
                            self.low_sell_timer += 1
                    else: # 價格或買點異常
                        trade_reward = -0.1 # 給一個小的固定懲罰
                # else: # 如果 buy_point 是 0 就不計算交易獎勵

                self.position -= sell_units
                if self.position < 1e-6: self.position = 0
                self.no_action_timer = 0
                trade_executed = True
            else:
                trade_ratio = 0 # 賣出單位太小，視為無效
                reward -= 0.01
        
        if not trade_executed:
            self.cumulative_invalid_trade_penalty -= 0.01

        # --- 組合獎勵 ---
        # 1. 交易盈虧獎勵
        reward += trade_reward
        self.cumulative_trade_reward += trade_reward # 追蹤交易逞罰

        # 2. 持有/活躍相關 (保持較低影響)
        HOLD_PENALTY_FACTOR = 0.1 # <<<=== **可調參數 3**
        HOLD_REWARD = 1      # <<<=== **可調參數 4** 取消
        if trade_ratio == 0 and self.no_action_timer > 15:
            hold_penalty = np.sqrt(self.no_action_timer - 15) * HOLD_PENALTY_FACTOR
            reward -= hold_penalty
            self.cumulative_hold_penalty -= hold_penalty

        # 3. 資產變化獎勵 (重要性提高)
        # 在計算 total_balance 後計算
        current_total_balance = self.balance + self.position * price # 先計算當前值
        ASSET_REWARD_WEIGHT = 3000 # <<<=== **可調參數 5** (提高權重)
        if previous_total_balance > 0 and current_total_balance > 0:
            asset_change_reward = np.log(current_total_balance / previous_total_balance) * ASSET_REWARD_WEIGHT
            reward += asset_change_reward
            self.cumulative_asset_change_reward += asset_change_reward
        # else: asset_change_reward = 0 # 初始或異常情況

        # 4. 回撤懲罰 (使用平方懲罰)
        # 在更新完 history 和 total_balance 後計算
        self.total_balance = current_total_balance # 更新狀態
        self.balance_history.append(self.total_balance) # 更新歷史
        # 更新峰值 (方法一，如果需要在 step 中使用)
        self.peak_balance = max(self.peak_balance, self.total_balance)

        current_drawdown = (self.peak_balance - self.total_balance) / self.peak_balance if self.peak_balance > 0 else 0
        DRAWDOWN_THRESHOLD = 0.20    # <<<=== **可調參數 6**
        DRAWDOWN_PENALTY_SCALE = 4 # <<<=== **可調參數 7** (使用平方懲罰的尺度)
        if current_drawdown > DRAWDOWN_THRESHOLD:
            exceeding_drawdown = current_drawdown - DRAWDOWN_THRESHOLD
            penalty = (exceeding_drawdown ** 2) * DRAWDOWN_PENALTY_SCALE 
            reward -= penalty
            self.cumulative_drawdown_penalty -= penalty
            

        # --- 檢查結束條件 ---
        self.current_step += 1
        is_end_of_data = self.current_step >= len(self.data)
        is_bankrupt = self.balance <= 5 and self.position <= 0
        done = is_end_of_data or is_bankrupt

        # --- 獲取下一步的歸一化狀態 ---
        next_normalized_state = self._get_normalized_state()

        # --- 最終獎勵調整 (可選) ---
        if done:
            # 計算最終的最大回撤
            self.max_drawdown = self.get_max_drawdown()
            # 可以在結束時給予基於最終資產的額外獎勵/懲罰
            FINAL_PNL_REWARD_SCALE = 100 # <<<=== **可調參數 8**
            final_pnl_reward = (self.total_balance - self.initial_balance) / self.initial_balance * FINAL_PNL_REWARD_SCALE
            reward += final_pnl_reward
            # 對破產給予懲罰
            if is_bankrupt and not is_end_of_data:
                reward -= 20 # <<<=== **可調參數 9**

        return next_normalized_state, reward, done, trade_executed # 返回歸一化狀態

    def get_max_drawdown(self):
        # ... (實現不變) ...
        if len(self.balance_history) < 2: return 0.0
        balance_array = np.array(self.balance_history)
        # 確保 cumulative_max 不為零
        cumulative_max = np.maximum.accumulate(balance_array)
        cumulative_max[cumulative_max == 0] = 1e-9 # 避免除以零
        drawdowns = (cumulative_max - balance_array) / cumulative_max
        # 處理可能的 NaN 或 Inf (雖然上面避免了除零，但以防萬一)
        drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        return max_drawdown
    
if __name__=="__main__":
    # 創建環境
    env = CryptoTradingEnv("technical/BTCUSDT_1d_technical.csv",data_split="train")

    # --- 擬合 Scaler ---
    print("開始收集狀態數據用於擬合 Scaler...")
    initial_states_for_scaler = []
    # 創建一個臨時環境專門用於數據收集，避免影響主環境狀態
    # 使用訓練數據集進行收集
    temp_env_for_scaler = CryptoTradingEnv("technical/BTCUSDT_1d_technical.csv", data_split="train")

    # 獲取初始狀態 (未歸一化)
    temp_state_unscaled = temp_env_for_scaler.get_current_unscaled_state_vector() # 使用新方法
    initial_states_for_scaler.append(temp_state_unscaled)

    done = False
    step_count = 0
    max_collect_steps = len(temp_env_for_scaler.data) - 1 # 跑完整個訓練數據集
    print(f"將在最多 {max_collect_steps} 步內收集狀態...")

    # 運行一個完整的 episode (或指定步數) 來收集不同狀態
    while not done and step_count < max_collect_steps:
        action = temp_env_for_scaler.action_space.sample() # 使用隨機動作探索
        _, _, done, _ = temp_env_for_scaler.step(action)   # 執行一步
        # 獲取當前步的未歸一化狀態
        next_state_unscaled = temp_env_for_scaler.get_current_unscaled_state_vector() # 使用新方法
        initial_states_for_scaler.append(next_state_unscaled)
        step_count += 1
        if step_count % 500 == 0:
            print(f"已收集 {step_count} 個狀態...")

    print(f"總共收集了 {len(initial_states_for_scaler)} 個狀態用於擬合 Scaler。")

    # 將收集到的狀態列表轉換為 NumPy 數組
    scaler_data = np.array(initial_states_for_scaler)

    # 確保收集到了數據並且維度正確
    if scaler_data.shape[0] > 0 and scaler_data.shape[1] == env.observation_space.shape[0]:
        print("正在擬合 Scaler...")
        # 使用收集到的數據擬合主訓練環境的 Scaler
        env.fit_scaler(scaler_data) # 調用環境的 fit_scaler 方法

        # *** 關鍵步驟：將擬合好的 Scaler 同步到測試環境 ***
        print("將擬合好的 Scaler 應用到測試環境...")
        env.scaler = env.scaler # 直接複製 scaler 對象
        env._scaler_fitted = True # 設置測試環境的標誌為 True

        print("Scaler 擬合完成並已應用於訓練和測試環境。")
    else:
        print(f"[錯誤] 收集到的數據維度不正確或沒有收集到數據！Shape: {scaler_data.shape}, Expected columns: {env.observation_space.shape[0]}")
        # 根據情況決定是否退出程序
        exit()

    # 清理臨時環境和數據，釋放內存
    del temp_env_for_scaler
    del initial_states_for_scaler
    del scaler_data
    print("臨時數據已清理。")

    # 重置環境並獲得初始狀態
    state = env.reset()
    print(f"Initial state: {state}")

    # 測試環境，進行幾個時間步
    for step in range(10):  # 測試前10個時間步
        # 隨機選擇一個動作（0: 持有, 1: 買入, 2: 賣出）
        action = int(input(f"第{step+1}輪選擇:"))

        # 執行動作
        next_state, reward, done, info = env.step(action)

        # 輸出結果
        print(f"Action: {action} -> {'Hold' if action == 0 else 'Buy' if action == 1 else 'Sell'}")
        print(f"Next state: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Current Balance: {env.balance:.2f},Coin: {env.position}, Total Balance: {env.total_balance:.2f}")
        # 若環境完成了，退出循環
        if done:
            print(f"Episode finished after {step+1} steps.")
            break