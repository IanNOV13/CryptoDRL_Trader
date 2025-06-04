import tensorflow as tf
from collections import deque
import numpy as np
from plotter import plot_segment
from utils import notify_discord_webhook

def evaluate_model(agent, eval_env, sequence_length, episode, eval_episodes=1):
    """在測試環境上評估訓練好的模型"""
    print(f"\n--- 開始使用{episode}的模型對測試集進行評估 ({eval_episodes} episodes)) ---")
    total_rewards = []
    final_balances = []
    max_drawdowns = []
    sharpe_ratio = []
    calmar_ratio = []
    peak_balances = []
    all_hight_sells = []
    all_low_sells = []
    all_steps = []

    for i in range(eval_episodes):
        eval_raw_state = eval_env.reset()
        eval_state_deque = deque(maxlen=sequence_length)
        for _ in range(sequence_length):
            eval_state_deque.append(eval_raw_state)
        eval_state_sequence = np.array(eval_state_deque)

        eval_done = False
        eval_total_reward = 0
        eval_t = 0
        
        while not eval_done:
            eval_t += 1
            eval_action = agent.select_action(eval_state_sequence)

            eval_next_raw_state, eval_reward, eval_done, eval_trade_executed = eval_env.step(eval_action)

            eval_state_deque.append(eval_next_raw_state)
            eval_next_state_sequence = np.array(eval_state_deque)

            eval_total_reward += eval_reward
            eval_state_sequence = eval_next_state_sequence

        plot_segment(episode, eval_env)

        # 記錄評估結果
        total_rewards.append(eval_total_reward)
        final_balances.append(eval_env.total_balance)
        max_drawdowns.append(eval_env.max_drawdown)
        sharpe_ratio.append(eval_env.sharpe_ratio)
        calmar_ratio.append(eval_env.calmar_ratio)
        peak_balances.append(np.max(eval_env.balance_history)) # 從歷史記錄計算峰值
        all_hight_sells.append(eval_env.hight_sell_timer)
        all_low_sells.append(eval_env.low_sell_timer)
        all_steps.append(eval_t)
        print(f"Eval Ep {i+1}: Steps={eval_t}, Reward={eval_total_reward:.2f}, Balance={eval_env.total_balance:.2f}, Peak={np.max(eval_env.balance_history):.2f}")
        print(f"Sharpe={eval_env.sharpe_ratio:.2f}, Calmar={eval_env.calmar_ratio:.2f}, MaxDrawdown={eval_env.max_drawdown*100:.2f}%")
        
    # 計算平均結果
    avg_reward = np.mean(total_rewards)
    avg_balance = np.mean(final_balances)
    avg_max_drawdown = np.mean(max_drawdowns)
    avg_sharpe_ratio = np.mean(sharpe_ratio)
    avg_calmar_ratio = np.mean(calmar_ratio)
    avg_peak = np.mean(peak_balances)
    
    report = f"--- Evaluation Summary (Avg over {eval_episodes} episodes) ---\n"
    report +=f"Avg Reward: {avg_reward:.2f}\n"
    report +=f"Avg Final Balance: {avg_balance:.2f}\n"
    report +=f"Avg Sharpe Ratio: {avg_sharpe_ratio:.2f}\n"
    report += f"Avg Calmar Ratio: {avg_calmar_ratio:.2f}\n"
    report += f"Avg Max Drawdown: {avg_max_drawdown*100:.2f}%\n"
    report +=f"Avg Peak Balance: {avg_peak:.2f}\n"
    report +=f"----------------------------------------------------\n"
    
    print(report)
    notify_discord_webhook(report)
