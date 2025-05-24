import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from utils import notify_discord_webhook
import pandas as pd

def plot_segment(episode, env):
    close_prices_all = env.df["close"].values
    buy_positions = env.buy_positions
    sell_positions = env.sell_positions
    mode = env.data_split
    date = pd.to_datetime(env.df["timestamp"].values).to_list()
    data = {
        "close_prices_all" : close_prices_all,
        "buy_positions" : buy_positions,
        "sell_positions" : sell_positions,
        "mode" : mode,
        "date" : date,
        "episode" : episode
    }
    actual_range = len(close_prices_all)
    steps_per_plot = actual_range // 100
    remaining_steps = actual_range % 100
    plot1_start_index = 0
    plot1_end_index = 100 + remaining_steps
    if plot1_end_index > plot1_start_index: # 確保範圍有效
        _plot_segment(1, plot1_start_index, plot1_end_index, f"(Steps {plot1_start_index}-{plot1_end_index-1})", data)
    # --- 計算並繪製第二部分 ---
    plot2_start_index = plot1_end_index 
    for i in range(2 ,steps_per_plot):
            plot2_end_index = plot1_end_index + i * 100 # 確保結束點正確
            if plot2_end_index > plot2_start_index: # 確保範圍有效
                _plot_segment(i+2, plot2_start_index, plot2_end_index, f"(Steps {plot2_start_index}-{plot2_end_index-1})", data)
            plot2_start_index = plot2_end_index

def _plot_segment(plot_part_num, start_idx, end_idx, title_suffix,data:dict): 
    plt.figure(figsize=(15, 7)) # 每個部分都創建一個新圖形
    plot_indices_segment = data['date'][start_idx:end_idx]#np.arange(start_idx, end_idx)
    close_prices_segment = data['close_prices_all'][start_idx:end_idx]

    plt.plot(plot_indices_segment, close_prices_segment, label=f'Price', color='blue', alpha=0.8)

    # 過濾買賣點到當前段 (buy_positions 和 sell_positions 存儲的是索引)
    buy_indices_in_segment = [idx_tuple[1] for idx_tuple in data['buy_positions'] if start_idx <= idx_tuple[1] < end_idx]
    sell_indices_in_segment = [idx_tuple[1] for idx_tuple in data['sell_positions'] if start_idx <= idx_tuple[1] < end_idx]
    
    if len(buy_indices_in_segment) > 0:
        buy_prices_in_segment = data['close_prices_all'][buy_indices_in_segment]
        # 買入點的 x 座標需要從 date 字典中根據索引獲取
        buy_x_coords = [data['date'][i] for i in buy_indices_in_segment]
        plt.scatter(buy_x_coords, buy_prices_in_segment, color='lime', marker='^', label='Buy', s=100, edgecolors='black')

    if len(sell_indices_in_segment) > 0:
        sell_prices_in_segment = data['close_prices_all'][sell_indices_in_segment]
        # 賣出點的 x 座標需要從 date 字典中根據索引獲取
        sell_x_coords = [data['date'][i] for i in sell_indices_in_segment] # <--- 修正這裡
        plt.scatter(sell_x_coords, sell_prices_in_segment, color='red', marker='v', label='Sell', s=100, edgecolors='black')

    # 繪製買入標記並顯示買入比例
    for (action_ratio, idx) in data['buy_positions']:
        if start_idx <= idx < end_idx:
            # 文本標記的 x 座標也需要從 date 字典中獲取
            text_x_coord = data['date'][idx] # <--- 修正這裡
            plt.text(text_x_coord, data['close_prices_all'][idx] * 1.01, f"{action_ratio}", fontsize=9, color="green", ha="center")
    # 繪製賣出標記並顯示賣出比例
    for (action_ratio, idx) in data['sell_positions']:
        if start_idx <= idx < end_idx:
            # 文本標記的 x 座標也需要從 date 字典中獲取
            text_x_coord = data['date'][idx] # <--- 修正這裡
            plt.text(text_x_coord, data['close_prices_all'][idx] * 0.99, f"{action_ratio}", fontsize=9, color="red", ha="center")

    plt.title(f'Episode {data["episode"]+1} - Trading Actions {title_suffix}')
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
    save_dir = f'.//training_plots//episode_{data["episode"]+1}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_filename_segment = os.path.join(save_dir, f"{data['mode']}_episode_{data['episode']+1}_part{plot_part_num}.png")
    plt.savefig(plot_filename_segment)
    plt.close() # 關閉當前圖形，釋放內存
    notify_discord_webhook(plot_filename_segment,"image")