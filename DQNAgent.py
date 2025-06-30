import numpy as np
import random
import tensorflow as tf
import joblib
import os
from collections import deque

# 假設你的模型構建函數在 model_builder.py (或 dqn_gru_model.py)
from dqn_lstm_model import build_inception_lstm_dqn_model

class DQNAgent:
    def __init__(self,
                 model_input_shape,
                 action_size,
                 data, # 字典，包含 LR, Epsilon 等調度所需參數
                 target_update_freq=1000, # 每多少步硬更新一次目標網絡
                 ):

        self.model_input_shape = model_input_shape
        self.action_size = action_size
        self.gamma = data.get("GAMMA", 0.95)
        self.batch_size = data.get("BATCH_SIZE", 64)
        self.model_path = data.get("MODEL_PATH", "dqn_model")
        self.memory_size = data.get("MEMORY_SIZE", 100000)

        # Epsilon 調度參數
        self.epsilon_initial = data.get("INITIAL_EPSILON", 1.0)
        self.epsilon_min_cycle = data.get("MIN_EPSILON_CYCLE", 0.05)
        self.epsilon_cycle_length = data.get("EPSILON_CYCLE_LENGTH", 1000) # 回合數
        self.epsilon_min_global = data.get("EPSILON_MIN", 0.01)
        self.epsilon = self.epsilon_initial # 當前 Epsilon

        # 學習率調度參數
        self.lr_initial = data.get("INITIAL_LR", 1e-4)
        self.lr_drop_rate = data.get("LR_DROP_RATE", 0.1)
        self.lr_epochs_drop = data.get("EPOCHS_DROP_RL", 10000) # 總步數
        self.lr_alpha = data.get("ALPHA_LR", 1e-6)
        self.current_lr = self.lr_initial # 當前學習率

        # 經驗回放池
        self.memory = deque(maxlen=self.memory_size)
        self.best_balance_memory = deque(maxlen=self.memory_size) # 或者與主 memory 不同大小
        self.best_reward_memory = deque(maxlen=self.memory_size)

        # 創建主網絡和目標網絡
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # 初始化時同步權重

        # 編譯後的預測函數
        self._compile_predict_functions()

        self.target_update_freq = target_update_freq

    def _build_model(self):
        # 創建優化器實例，學習率會在訓練循環中動態設置
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_initial) # 初始學習率
        model = build_inception_lstm_dqn_model(self.model_input_shape, self.action_size)
        model.compile(optimizer=optimizer, loss='mse') # 使用 MSE 損失
        return model

    def _compile_predict_functions(self):
        """編譯用於加速預測的 TensorFlow Functions"""
        @tf.function(input_signature=[tf.TensorSpec(shape=(1,) + self.model_input_shape, dtype=tf.float32)])
        def compiled_predict_single(state_input):
            return self.model(state_input, training=False)
        self._compiled_predict_single = compiled_predict_single

        @tf.function(input_signature=[tf.TensorSpec(shape=(None,) + self.model_input_shape, dtype=tf.float32)])
        def compiled_predict_batch_main(state_batch_input):
            return self.model(state_batch_input, training=False)
        self._compiled_predict_batch_main = compiled_predict_batch_main

        @tf.function(input_signature=[tf.TensorSpec(shape=(None,) + self.model_input_shape, dtype=tf.float32)])
        def compiled_predict_batch_target(state_batch_input):
            return self.target_model(state_batch_input, training=False)
        self._compiled_predict_batch_target = compiled_predict_batch_target

    def update_target_model(self, hard_update=True, tau=0.001):
        """更新目標網絡的權重"""
        if hard_update:
            self.target_model.set_weights(self.model.get_weights())
        else: # Soft update
            target_weights = self.target_model.get_weights()
            online_weights = self.model.get_weights()
            new_weights = [tau * online + (1.0 - tau) * target for online, target in zip(online_weights, target_weights)]
            self.target_model.set_weights(new_weights)

    def update_target_model_if_needed(self, global_step_count, hard_update=True):
        """根據全局步數決定是否更新目標網絡 (用於硬更新)"""
        if hard_update and global_step_count % self.target_update_freq == 0:
            #print(f"\n--- 更新目標網絡權重 (步數: {global_step_count}) ---")
            self.update_target_model(hard_update=True)
            return True
        return False


    def select_action(self, state_sequence):
        """根據 Epsilon-greedy 策略選擇動作"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # state_sequence 應該是 (SEQUENCE_LENGTH, num_features)
            # 需要 reshape 成 (1, SEQUENCE_LENGTH, num_features)
            model_input_state = state_sequence.reshape((1,) + self.model_input_shape)
            q_values_tensor = self._compiled_predict_single(tf.convert_to_tensor(model_input_state, dtype=tf.float32))
            return np.argmax(q_values_tensor.numpy()[0])

    def store_experience(self, state, action, reward, next_state, done):
        """存儲經驗到主經驗池"""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)


    def update_best_memory_if_needed(self, current_episode_experiences, performance_value, best_performance_so_far, memory_to_update):
        """如果當前回合表現更好，則更新對應的最佳經驗池"""
        if performance_value > best_performance_so_far:
            # print(f"更新 {memory_to_update}，新最佳表現: {performance_value:.2f}") # 日誌可以更詳細
            memory_to_update.clear()
            for exp in current_episode_experiences:
                memory_to_update.append(exp)
            return performance_value, True # 返回新的最佳表現值
        return best_performance_so_far, False

    def _get_sample_batch(self):
        """執行混合抽樣邏輯"""
        minibatch = []
        # 基本條件：主池有足夠數據
        if len(self.memory) < self.batch_size // 2: # 至少需要主池的一半，或者一個更小的閾值
            return None # 樣本不足，無法訓練

        can_mixed_sample = (len(self.best_balance_memory) > 0 or len(self.best_reward_memory) > 0)

        if can_mixed_sample:
            main_target_size = int(self.batch_size * 0.9)
            remaining_size = self.batch_size - main_target_size
            balance_target_size = int(remaining_size * 0.8)
            reward_target_size = remaining_size - balance_target_size

            if len(self.memory) < main_target_size:
                main_target_size = len(self.memory)
                remaining_size = self.batch_size - main_target_size
                balance_target_size = int(remaining_size * 0.8)
                reward_target_size = remaining_size - balance_target_size

            main_minibatch = random.sample(self.memory, main_target_size)
            balance_batch = random.choices(list(self.best_balance_memory), k=balance_target_size) if len(self.best_balance_memory) > 0 else []
            reward_batch = random.choices(list(self.best_reward_memory), k=reward_target_size) if len(self.best_reward_memory) > 0 else []

            minibatch.extend(main_minibatch)
            minibatch.extend(balance_batch)
            minibatch.extend(reward_batch)

            missing_count = self.batch_size - len(minibatch)
            if missing_count > 0:
                potential_supplements = [exp for exp in self.memory if exp not in main_minibatch]
                actual_missing_count = min(missing_count, len(potential_supplements))
                if actual_missing_count > 0:
                    minibatch.extend(random.sample(potential_supplements, actual_missing_count))
        else: # 只能從主池抽樣
            if len(self.memory) >= self.batch_size:
                minibatch = random.sample(self.memory, self.batch_size)
            else:
                return None # 主池也不足

        if minibatch:
            random.shuffle(minibatch)
        return minibatch


    def can_train(self):
        """判斷是否滿足訓練的基本條件 (例如，主經驗池大小)"""
        return len(self.memory) >= self.batch_size

    def train_batch(self):
        """從經驗池抽樣並訓練模型 (DDQN)"""
        minibatch = self._get_sample_batch()
        if not minibatch or len(minibatch) < self.batch_size // 2: # 如果抽樣失敗或樣本太少
            # print("樣本不足，跳過此次訓練")
            return False # 指示訓練未執行

        actual_batch_size = len(minibatch)

        state_sequences_batch = np.array([transition[0] for transition in minibatch])
        actions_batch = np.array([transition[1] for transition in minibatch])
        rewards_batch = np.array([transition[2] for transition in minibatch])
        next_state_sequences_batch = np.array([transition[3] for transition in minibatch])
        dones_batch = np.array([transition[4] for transition in minibatch])

        # DDQN Target 計算
        q_values_current = self._compiled_predict_batch_main(tf.convert_to_tensor(state_sequences_batch, dtype=tf.float32)).numpy()
        q_values_next_online = self._compiled_predict_batch_main(tf.convert_to_tensor(next_state_sequences_batch, dtype=tf.float32)).numpy()
        q_values_next_target = self._compiled_predict_batch_target(tf.convert_to_tensor(next_state_sequences_batch, dtype=tf.float32)).numpy()

        targets = np.copy(q_values_current)
        for i in range(actual_batch_size):
            if dones_batch[i]:
                targets[i, actions_batch[i]] = rewards_batch[i]
            else:
                action_max_online = np.argmax(q_values_next_online[i])
                targets[i, actions_batch[i]] = rewards_batch[i] + self.gamma * q_values_next_target[i, action_max_online]

        history = self.model.fit(state_sequences_batch, targets, epochs=1, verbose=0, batch_size=actual_batch_size, shuffle=False) # shuffle=False 因為已手動打亂
        current_batch_loss = history.history['loss'][0]
        return current_batch_loss

    def get_current_epsilon(self, episode_num):
        """計算當前回合的 Epsilon (週期性餘弦退火)"""
        progress_in_cycle = (episode_num % self.epsilon_cycle_length) / float(self.epsilon_cycle_length)
        cosine_decay_epsilon = 0.5 * (1 + np.cos(np.pi * progress_in_cycle))
        current_epsilon_dynamic = (self.epsilon_initial - self.epsilon_min_cycle) * cosine_decay_epsilon + self.epsilon_min_cycle
        self.epsilon = max(current_epsilon_dynamic, self.epsilon_min_global)
        return self.epsilon

    def get_current_lr(self, global_step_count):
        """計算當前全局步數的學習率 (餘弦退火)"""
        cosine_decay = 0.5 * (1 + np.cos(np.pi * (global_step_count % self.lr_epochs_drop) / self.lr_epochs_drop))
        decayed = (1 - self.lr_drop_rate) * cosine_decay + self.lr_drop_rate
        new_lr = self.lr_initial * decayed
        self.current_lr = max(new_lr, self.lr_alpha)
        return self.current_lr

    def set_optimizer_lr(self, learning_rate):
        """設置優化器的學習率"""
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)

    def save_model(self, path_prefix):
        """保存主模型和目標模型"""
        self.model.save(path_prefix, save_format='tf')
        self.target_model.save(path_prefix + "_target", save_format='tf')
        print(f"模型已保存到: {path_prefix} (和 _target)")
        # --- 保存經驗池 ---
        try:
            # 為每個經驗池創建單獨的文件名
            main_memory_path = os.path.join(path_prefix, "main_memory.joblib")
            best_balance_memory_path = os.path.join(path_prefix, "best_balance_memory.joblib")
            best_reward_memory_path = os.path.join(path_prefix, "best_reward_memory.joblib")

            joblib.dump(self.memory, main_memory_path)
            print(f"主經驗池已保存到: {main_memory_path}")

            joblib.dump(self.best_balance_memory, best_balance_memory_path)
            print(f"最佳餘額經驗池已保存到: {best_balance_memory_path}")

            joblib.dump(self.best_reward_memory, best_reward_memory_path)
            print(f"最佳獎勵經驗池已保存到: {best_reward_memory_path}")

        except Exception as e:
            print(f"[錯誤] 保存經驗池失敗: {e}")


    def load_model(self, path_prefix):
        """加載主模型和目標模型"""
        main_model_loaded = False
        target_model_loaded = False
        try:
            self.model = tf.keras.models.load_model(path_prefix, compile=False)
            main_model_loaded = True
            print(f"主模型已從 {path_prefix} 加載。")
            try:
                self.target_model = tf.keras.models.load_model(path_prefix + "_target", compile=False)
                target_model_loaded = True
                print(f"目標模型已從 {path_prefix}_target 加載。")
            except Exception as e_target: # 捕獲目標模型加載失敗
                #print(f"加載目標模型失敗 ({path_prefix}_target): {e_target}")
                print("將從主網絡複製權重到目標網絡。")
                # 如果主模型已加載，目標模型可以從主模型複製
                if main_model_loaded:
                    # self.target_model = self._build_model() # 不需要重新 build，直接複製結構
                    self.target_model.set_weights(self.model.get_weights())
                else: # 如果主模型也沒加載，則兩個都重新 build
                    #print("主模型也未加載，重新構建主模型和目標模型。")
                    self.model = self._build_model()
                    self.target_model = self._build_model()
                    self.update_target_model() # 同步權重

        except Exception as e_main: # 捕獲主模型加載失敗
            #print(f"加載主模型失敗 ({path_prefix}): {e_main}")
            #print("重新構建主模型和目標模型。")
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

        finally:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.current_lr)
            self.model.compile(optimizer=optimizer, loss='mse')
            self._compile_predict_functions()

        # --- 加載經驗池 ---
            try:
                main_memory_path = os.path.join(path_prefix, "main_memory.joblib")
                best_balance_memory_path = os.path.join(path_prefix, "best_balance_memory.joblib")
                best_reward_memory_path = os.path.join(path_prefix, "best_reward_memory.joblib")

                if os.path.exists(main_memory_path):
                    self.memory = joblib.load(main_memory_path)
                    print(f"主經驗池已從 {main_memory_path} 加載。")
                else:
                    print(f"主經驗池文件未找到: {main_memory_path}")

                if os.path.exists(best_balance_memory_path):
                    self.best_balance_memory = joblib.load(best_balance_memory_path)
                    print(f"最佳餘額經驗池已從 {best_balance_memory_path} 加載。")
                else:
                    print(f"最佳餘額經驗池文件未找到: {best_balance_memory_path}")

                if os.path.exists(best_reward_memory_path):
                    self.best_reward_memory = joblib.load(best_reward_memory_path)
                    print(f"最佳獎勵經驗池已從 {best_reward_memory_path} 加載。")
                else:
                    print(f"最佳獎勵經驗池文件未找到: {best_reward_memory_path}")

            except Exception as e_memory:
                print(f"[錯誤] 加載經驗池失敗: {e_memory}")

        return main_model_loaded, target_model_loaded
