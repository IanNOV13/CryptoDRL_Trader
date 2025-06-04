import tensorflow as tf
import numpy as np
import os
import sys
import psutil
import signal
import datetime
from collections import deque
from dotenv import load_dotenv
import gc

# 導入自定義模塊
from config_loader import ConfigLoader # 假設你創建了這個模塊
from data_handler import prepare_data_and_env # 假設你創建了這個函數來處理數據和創建環境
from DQNAgent import DQNAgent # 核心智能體
from evaluator import evaluate_model
from plotter import plot_segment
from utils import notify_discord_webhook, clear_lines, setup_gpu, set_mixed_precision_policy
save_on_exit_tag = False
load_dotenv()

# --- 建立Ctrl+C 或系統送出終止訊號的保存動作 ---
def save_on_exit(signal_received, frame):
    global save_on_exit_tag
    save_on_exit_tag = not(save_on_exit_tag)
    notify_discord_webhook(f"中斷訊號更改成{save_on_exit_tag}。")
    

signal.signal(signal.SIGINT, save_on_exit)
signal.signal(signal.SIGTERM, save_on_exit)

def main():
    # 1. 加載配置
    config = ConfigLoader("setting.json").load_settings()
    setup_gpu() # GPU 初始化
    set_mixed_precision_policy() #啟用混和精度訓練

    # 2. 準備數據和環境
    #    這個函數內部可能處理數據路徑、創建 CryptoTradingEnv 實例、處理 Scaler
    train_env, test_env, scaler = prepare_data_and_env(config)

    # 獲取環境信息
    num_features = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n
    model_input_shape = (config['SEQUENCE_LENGTH'], num_features)
    print(model_input_shape)
    # 3. 初始化 Agent
    agent = DQNAgent(
        model_input_shape = model_input_shape,
        action_size = action_size,
        data = config
    )

    # 4. 嘗試加載模型和進度 (或由 Agent 內部處理)
    start_episode = config["START_EPISODE"]
    best_reward = config["BEST_REWARD"]
    best_balance = config["BEST_BALANCE"]
    best_calmar_ratio = config["BEST_CALMAR_RATIO"]

    if os.path.exists(config["LOAD_MODEL_PATH"]):
        agent.load_model(config["LOAD_MODEL_PATH"])
        print("從檢查點繼續訓練...")
    else:
        print("開始新的訓練...")

    # 5. 使用一個 deque 來高效地維護狀態序列
    state_sequence_deque = deque(maxlen=config["SEQUENCE_LENGTH"])

    # 6. 主訓練循環
    global_step_count = config.get("global_step_count", 0)
    # --- 獲取當前進程對象 ---
    current_process = psutil.Process(os.getpid())

    for episode in range(start_episode, config["EPISODES"]):
        current_episode_experiences = []
        t = 0
        done = False
        new_calmar_ratio = False
        raw_state = train_env.reset()
        state_sequence_deque.clear()

        # --- 初始化 state_sequence ---
        for _ in range(config["SEQUENCE_LENGTH"]):
            state_sequence_deque.append(raw_state)
        # --- 初始化結束 ---

        # 更新 探索率 和學習率
        current_epsilon = agent.get_current_epsilon(episode)
        current_lr = agent.get_current_lr(global_step_count)
        agent.set_optimizer_lr(current_lr)

        total_reward_episode = 0 # 單回合總獎勵統計
        total_loss = 0
        loss = 0

        while not done:
            # --- 獲取動作 ---
            # 將 deque 轉換為 NumPy 數組，作為初始的狀態序列
            state_sequence = np.array(state_sequence_deque)
            action = agent.select_action(state_sequence)
            next_state_sequence, reward_step, done, trade_executed = train_env.step(action)
            # --- 保存當前經驗並更新 並更新 state_sequence ---
            state_sequence_deque.append(next_state_sequence)
            next_state_sequence = np.array(state_sequence_deque)
            agent.store_experience(state_sequence, action, reward_step, next_state_sequence, done)
            current_episode_experiences.append((state_sequence, action, reward_step, next_state_sequence, done))
            
            total_reward_episode += reward_step
            state_sequence = next_state_sequence
            global_step_count += 1
            t += 1

            # 訓練 (每隔7步)
            if global_step_count % 7 == 0:
                if agent.can_train(): # agent 內部判斷經驗池大小等
                    loss = agent.train_batch() # 內部處理混合抽樣、DDQN Target、model.fit
                    total_loss = (total_loss + loss) / 2

                # 更新目標網絡
                agent.update_target_model_if_needed(global_step_count)
            
            if t > 1:
                clear_lines(9)
            # 獲取當前進程使用的物理內存 (RSS - Resident Set Size)
            mem_info = current_process.memory_info()
            rss_gb = mem_info.rss / (1024**3) # 轉換為 GB
            #顯示目前回合的即時訓練成績
            print(f"🚅{datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')} 中斷訊號：{save_on_exit_tag}")
            print(f"回合{episode}/{config['EPISODES']}，探索機率：{current_epsilon:.4f}，學習率： {current_lr:.6f}，loss： {loss:6.2f}/{total_loss:6.2f}")
            print(f"步數：{t}/{config['BEST_STEPS']}")
            print(f"總獎勵：{total_reward_episode:.2f}/{best_reward:6.2f}")
            print(f"交易分數：{train_env.cumulative_trade_reward:6.2f}，資產變化分數：{train_env.cumulative_asset_change_reward:6.2f}，持有與不動作分數：{train_env.cumulative_hold_penalty:6.2f}，回徹逞罰分數：{train_env.cumulative_drawdown_penalty:6.2f}，無效交易懲罰：{train_env.cumulative_invalid_trade_penalty:6.2f}")
            print(f"高賣次數：{train_env.hight_sell_timer}，低賣次數：{train_env.low_sell_timer}，比例{(train_env.hight_sell_timer/max(train_env.low_sell_timer,1)):6.2f}")
            print(f"RAM：{rss_gb:.2f}/{config['MEMORY_LIMIT_GB']}")
            print(f"餘額：{train_env.total_balance:6.2f}/{best_balance:6.2f}，餘額歷史最高值：{np.max(train_env.balance_history):6.2f}")
            print(f"最大回撤：{train_env.max_drawdown:6.2f}，夏普比率：{train_env.sharpe_ratio:6.2f}，卡爾馬比率：{train_env.calmar_ratio:6.2f}/{best_calmar_ratio:6.2f}")
        # --- 回合結束 ---
        # --- 更新最佳經驗池 ---
        best_reward, new_reward = agent.update_best_memory_if_needed(current_episode_experiences, total_reward_episode, best_reward, agent.best_reward_memory)
        best_balance, new_balance = agent.update_best_memory_if_needed(current_episode_experiences, train_env.total_balance, best_balance, agent.best_balance_memory)
        if train_env.calmar_ratio > best_calmar_ratio:
            best_calmar_ratio = train_env.calmar_ratio
            new_calmar_ratio = True
        # --- 顯示本回合最終資訊，並傳送到discord ---
        clear_lines(9)
        report_text = ""
        report_text += f"🚩{datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_text += f"回合{episode}/{config['EPISODES']}，探索機率：{current_epsilon:.4f}，學習率: {current_lr:.6f}，Avg loss：{total_loss:6.2f}\n"
        report_text += f"步數：{t}/{config['BEST_STEPS']}\n"
        report_text += f"總獎勵：{total_reward_episode:.2f}/{best_reward:6.2f}\n"
        report_text += f"交易分數：{train_env.cumulative_trade_reward:6.2f}，資產變化分數：{train_env.cumulative_asset_change_reward:6.2f}，持有與不動作分數：{train_env.cumulative_hold_penalty:6.2f}，回徹逞罰分數：{train_env.cumulative_drawdown_penalty:6.2f}，無效交易懲罰：{train_env.cumulative_invalid_trade_penalty:6.2f}\n"
        report_text += f"高賣次數：{train_env.hight_sell_timer}，低賣次數：{train_env.low_sell_timer}，比例{(train_env.hight_sell_timer/max(train_env.low_sell_timer,1)):6.2f}\n"
        report_text += f"RAM：{rss_gb:.2f}/{config['MEMORY_LIMIT_GB']}\n"
        report_text += f"餘額：{train_env.total_balance:6.2f}/{best_balance:6.2f}，餘額歷史最高值：{np.max(train_env.balance_history):6.2f}\n"
        report_text += f"最大回撤：{train_env.max_drawdown:6.2f}，夏普比率：{train_env.sharpe_ratio:6.2f}，卡爾馬比率：{train_env.calmar_ratio:6.2f}/{best_calmar_ratio:6.2f}\n"
        print(report_text)
        notify_discord_webhook(report_text)
        # --- 繪圖與保存模型、進度 ---
        if new_balance:
            agent.save_model(config["MODEL_PATH"] + "_BEST_BALANCE")
            notify_discord_webhook("GET NEW BEST BALANCE 🎉")
        if new_reward:
            agent.save_model(config["MODEL_PATH"] + "_BEST_REWARD")
            notify_discord_webhook("GET NEW BEST REWARD 🎉")
        if new_calmar_ratio:
            agent.save_model(config["MODEL_PATH"] + "_BEST_CALMAR_RATIO")
            notify_discord_webhook("GET NEW BEST CALMAR RATIO 🎉")
        if episode%50 == 0:
            agent.save_model(config["MODEL_PATH"])
        
        if new_balance or new_reward or new_calmar_ratio or episode%25 == 0 or save_on_exit_tag:
            plot_segment(episode, train_env)
            evaluate_model(agent, test_env, model_input_shape[0], episode)
            ConfigLoader("setting.json").save_progress(
                episode = episode,
                best_reward = best_reward,
                best_balance = best_balance,
                best_calmar_ratio = best_calmar_ratio,
                current_epsilon = current_epsilon,
                current_lr = current_lr,
                global_step_count = global_step_count
            )

            if save_on_exit_tag:
                agent.save_model(config["MODEL_PATH"])
                notify_discord_webhook("模型與訓練進度已保存完成。")
                sys.exit(0)
        # --- 記憶體清理 ---
        if rss_gb > config['MEMORY_LIMIT_GB']:
            agent.save_model(config["MODEL_PATH"])
            tf.keras.backend.clear_session()
            gc.collect()
            agent.load_model(config["MODEL_PATH"])
    # --- 訓練結束 ---
    agent.save_model(os.path.join(config["MODEL_PATH"], "final_model"))
    print("訓練完成。")

if __name__ == "__main__":
    main()