import json
import numpy as np
import os

class ConfigLoader:
    def __init__(self, settings_file="setting.json"):
        """
        初始化配置加載器。

        Args:
            settings_file (str): 設定檔的路徑。
        """
        self.settings_file = settings_file
        self.settings = self._load_settings()
        self._process_special_values()

    def _load_settings(self):
        """
        從 JSON 文件加載設定。如果文件不存在或格式錯誤，則返回一個包含預設值的字典。
        """
        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                settings = json.load(f)
                print(f"成功從 '{self.settings_file}' 加載設定。")
                return settings
        except FileNotFoundError:
            print(f"[警告] 設定檔 '{self.settings_file}' 未找到。將使用預設配置 (如果定義了)。")
            return self._get_default_settings()
        except json.JSONDecodeError:
            print(f"[錯誤] 設定檔 '{self.settings_file}' 格式錯誤。將使用預設配置 (如果定義了)。")
            return self._get_default_settings()
        except Exception as e:
            print(f"[錯誤] 加載設定檔時發生未知錯誤: {e}。將使用預設配置 (如果定義了)。")
            return self._get_default_settings()

    def _get_default_settings(self):
        """
        提供一組預設的配置參數。
        這在 setting.json 丟失或損壞時非常有用。
        你需要根據你的項目填充這些預設值。
        """
        print("正在使用預設配置參數。")
        return {
            "LOAD_MODEL_PATH": "BTC_1d_model_DEFAULT",
            "MODEL_PATH": "BTC_1d_model_DEFAULT",
            "EPISODES": 1000, # 預設總訓練回合
            "BATCH_SIZE": 256,
            "GAMMA": 0.95,
            "INITIAL_EPSILON": 1.0,
            "MIN_EPSILON_CYCLE": 0.05,
            "EPSILON_CYCLE_LENGTH": 500,
            "EPSILON_MIN": 0.01,
            "MEMORY_SIZE": 10000,
            "INITIAL_LR": 1e-4,
            "LR_DROP_RATE": 0.1,
            "EPOCHS_DROP_RL": 100000,
            "ALPHA_LR": 1e-6,
            "TIME_RANGE": 1000,
            "NUM_PLOTS": 4,
            "START_EPISODE": 0,
            "BEST_STEPS": 0,
            "BEST_REWARD": None, # 將在 _process_special_values 中處理
            "BEST_BALANCE": None, # 將在 _process_special_values 中處理
            "SEQUENCE_LENGTH": 30,
            "MEMORY_LIMIT_GB": 12.0,
            "USE_LINE_BOT": False,
            "TRAINING_INTERVAL": 3, # 示例：每隔多少步訓練一次
            "UPDATE_TARGET_EVERY": 1000, # 示例：每隔多少步更新目標網絡
            # --- 根據你的 setting.json 添加更多預設值 ---
        }

    def _process_special_values(self):
        """
        處理一些特殊的配置值，例如將 None 轉換為 -np.inf。
        """
        if self.settings.get("BEST_REWARD") is None:
            self.settings["BEST_REWARD"] = -np.inf
        if self.settings.get("BEST_BALANCE") is None:
            self.settings["BEST_BALANCE"] = -np.inf

        # 確保 SCALER_PATH 是基於 MODEL_PATH 生成的
        model_path = self.get("MODEL_PATH", "default_model_path") # 使用 get 避免 MODEL_PATH 也不存在
        self.settings["SCALER_PATH"] = os.path.join(model_path, "scaler.joblib")


    def get(self, key, default=None):
        """
        獲取指定鍵的配置值。

        Args:
            key (str): 配置項的鍵。
            default: 如果鍵不存在時返回的默認值。

        Returns:
            配置值或默認值。
        """
        return self.settings.get(key, default)

    def load_settings(self):
        """
        返回所有配置項的字典。
        """
        return self.settings

    def save_progress(self, episode, best_reward, best_balance, current_epsilon, current_lr, global_step_count):
        """
        保存訓練進度到 setting.json 文件。
        會先讀取現有設置，然後更新進度相關的鍵，再寫回。

        Args:
            episode (int): 當前完成的回合數。
            best_reward (float): 當前的最佳獎勵。
            best_balance (float): 當前的最佳餘額。
            current_epsilon (float): 當前的 Epsilon 值。
            current_lr (float): 當前的學習率。
            global_step_count (int): 當前的全局步數。
        """
        progress_to_save = {
            "START_EPISODE": episode + 1, # 下次從下一個回合開始
            "BEST_REWARD": float(best_reward) if best_reward != -np.inf else None, # 存儲 None 而不是 -inf
            "BEST_BALANCE": float(best_balance) if best_balance != -np.inf else None, # 存儲 None 而不是 -inf
            "INITIAL_EPSILON": float(current_epsilon), # 下次啟動時的初始 Epsilon
            "INITIAL_LR": float(current_lr),           # 下次啟動時的初始學習率 (如果 LR 調度是基於此)
            "global_step_count": int(global_step_count) # 保存全局步數
            # 可以根據需要添加更多需要保存的進度信息
        }

        # 讀取現有設置（或預設），然後更新
        current_config_on_disk = {}
        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                current_config_on_disk = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"讀取 '{self.settings_file}' 失敗或文件不存在/損壞，將基於預設和當前進度創建。")
            current_config_on_disk = self._get_default_settings() # 如果文件有問題，從預設開始

        # 更新進度信息，同時保留其他配置
        current_config_on_disk.update(progress_to_save)
        self.settings.update(progress_to_save) # 也更新內存中的 settings

        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(current_config_on_disk, f, indent=4)
            print(f"訓練進度已保存到 '{self.settings_file}'")
        except Exception as e:
            print(f"[錯誤] 保存訓練進度到 '{self.settings_file}' 失敗: {e}")

if __name__ == "__main__":
    # 創建配置加載器實例
    config_loader = ConfigLoader("setting_example.json") # 假設你的設定檔名

    # 獲取單個配置項
    episodes = config_loader.get("EPISODES", 100) # 如果 EPISODES 不存在，則默認為 100
    print(f"總回合數: {episodes}")
    print(f"最佳獎勵 (加載時): {config_loader.get('BEST_REWARD')}")
    print(f"Scaler 路徑: {config_loader.get('SCALER_PATH')}")


    # 獲取所有配置
    all_config = config_loader.load_settings()
    print("\n所有配置:")
    for key, value in all_config.items():
        print(f"  {key}: {value}")

    # 模擬保存進度
    print("\n模擬保存進度...")
    config_loader.save_progress(
        episode=100,
        best_reward=15000.0,
        best_balance=200000.0,
        best_steps=2200,
        current_epsilon=0.1,
        current_lr=0.00005,
        global_step_count=220000
    )
    print(f"最佳獎勵 (保存後，從內存讀取): {config_loader.get('BEST_REWARD')}")
    print(f"下次開始回合 (保存後，從內存讀取): {config_loader.get('START_EPISODE')}")

    # 再次加載檢查文件是否更新
    print("\n再次加載檢查文件...")
    config_loader_reloaded = ConfigLoader("setting_example.json")
    print(f"最佳獎勵 (重新加載後): {config_loader_reloaded.get('BEST_REWARD')}")
    print(f"下次開始回合 (重新加載後): {config_loader_reloaded.get('START_EPISODE')}")