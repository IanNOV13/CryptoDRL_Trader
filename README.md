# 基於深度強化學習的加密貨幣交易機器人

此項目是使用深度強化學習 (DRL) 技術，特別是雙重深度Q網絡 (DDQN) 算法，訓練一個能夠在加密貨幣市場（以 BTCUSDT 為例）進行自動化交易的智能體。

## 項目特點

*   **先進的DRL算法**：採用 DDQN 算法，減少Q值過高估計問題。
*   **複雜的狀態表示**：結合市場原始數據（OHLCV）、多種技術指標以及實時賬戶狀態（持倉比例、總資金相對初始本金比例、不活躍時長、相對成本）構建強化學習的狀態空間。
*   **動態獎勵函數**：精心設計的獎勵函數，旨在平衡盈利能力、風險控制和交易活躍度，並引入動態調整的交易獎勵尺度以抑制高頻刷分行為。
*   **靈活的超參數配置**：通過 `setting.json` 文件集中管理訓練參數，方便調優和斷點續訓。
*   **穩健的數據預處理**：使用 `StandardScaler` (或可選的 `RobustScaler`) 進行狀態歸一化，並實現了數據的自動加載、擬合和保存。
*   **詳細的訓練監控與報告**：
    *   實時在終端打印訓練進度，包含獎勵分解、交易統計等詳細信息。
    *   可選通過 Discord (或 LINE Bot) 推送每回合的訓練報告和評估結果。
    *   定期繪製交易行為圖表並保存。
*   **斷點續訓與進度保存**：
    *   能夠從上次中斷的地方繼續訓練。
    *   通過捕獲系統中斷信號 (Ctrl+C, SIGTERM) 安全保存模型和訓練進度到 `setting.json`。
*   **多經驗池混合抽樣**：結合主經驗池、最佳餘額回合經驗池和最佳獎勵回合經驗池進行抽樣訓練，強化對成功經驗的學習。
*   **先進的學習率與探索率調度**：
    *   學習率採用餘弦退火調度。
    *   探索率 (Epsilon) 採用週期性餘弦退火調度，鼓勵在訓練過程中進行重新探索。
*   **模塊化代碼結構**：環境 (`trading_env.py`)、模型 (`dqn_gru_model.py`) 和訓練腳本 (`train.py`) 分離，易於維護和擴展。

## 環境要求

*   Python 3.x
*   TensorFlow 2.x
*   Gymnasium (或 Gym)
*   NumPy
*   Pandas
*   Pandas-TA (用於計算技術指標)
*   Matplotlib (用於繪圖)
*   Scikit-learn (用於 StandardScaler/RobustScaler)
*   python-dotenv (用於加載環境變量)
*   requests (用於 Discord Webhook)
*   joblib (用於保存/加載 Scaler)
*   (可選) line-bot-sdk (v3) (如果使用 LINE Bot 通知)
*   (可選) keyboard (如果需要手動鍵盤暫停功能，並注意潛在的權限問題)

建議使用虛擬環境管理依賴：
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 項目結構

```
.
├── training_plots/          # 存放訓練過程中生成的圖表
├── technical/               # 存放計算了技術指標的數據文件
├── full_csv/                # 存放原始的市場數據 CSV 文件
├── BTC_1d_model_TEST/       # 模型保存目錄 (根據 setting.json 中的 MODEL_PATH)
│   ├── scaler.joblib        # 保存的 Scaler 對象
├── BEST_BALANCE/            # 最佳餘額模型
├── BEST_REWARD/             # 最佳獎勵模型
├── BEST_STEP/               # 最佳步數模型
├── trading_env.py           # 自定義的 Gym 交易環境
├── dqn_gru_model.py         # DDQN 模型架構定義
├── train_g.py               # 主訓練腳本
├── creat_technical_data.py  # 生成技術指標數據的腳本 (假設你有這個)
├── setting.json             # 存儲訓練參數和進度的配置文件
├── .env                     # 存儲敏感信息 (如 API Token, Webhook URL) - **不要提交到 Git**
└── README.md                # 本文件
```

## 使用說明

### 1. 準備數據

1.  將原始的加密貨幣市場數據 (OHLCV 格式) 保存為 CSV 文件，放置於 `full_csv/` 目錄下，例如 `full_csv/BTCUSDT_1d_full.csv`。
2.  (可選，但推薦) 運行 `creat_technical_data.py` 腳本來為原始數據計算技術指標，並將結果保存到 `technical/` 目錄下。
    ```bash
    python creat_technical_data.py --symbol BTCUSDT --interval 1d
    ```
    確保 `trading_env.py` 中的 `data_file` 參數指向這個包含技術指標的文件。

### 2. 配置環境變量

創建一個 `.env` 文件，並填寫必要的敏感信息，例如：

```
DISCORD_WEBHOOK="你的Discord_Webhook_URL"
LINE_CHANNEL_ACCESS_TOKEN="你的LINE_Channel_Access_Token"
USER_ID="你的LINE_User_ID"
```

### 3. 配置訓練參數

編輯 `setting.json` 文件，根據你的需求修改訓練參數，例如：

```json
{
    "LOAD_MODEL_PATH": "BTC_1d_model_TEST", // 或者特定最佳模型的路徑，如 "BTC_1d_model_TEST_BEST_BALANCE"
    "MODEL_PATH": "BTC_1d_model_TEST",
    "EPISODES": 8000,
    "BATCH_SIZE": 384,
    "GAMMA": 0.95,
    "INITIAL_EPSILON": 0.3,         // 如果是續訓，這個值可能來自上次保存
    "MIN_EPSILON_CYCLE": 0.05,
    "EPSILON_CYCLE_LENGTH": 1000,   // Epsilon 熱重啟週期
    "EPSILON_MIN": 0.01,
    "MEMORY_SIZE": 15000,
    "INITIAL_LR": 5e-05,
    "LR_DROP_RATE": 0.1,
    "EPOCHS_DROP_RL": 100000,       // 學習率退火週期 (全局步數)
    "ALPHA_LR": 1e-06,
    "TIME_RANGE": 1000,
    "NUM_PLOTS": 4,                 // 調整繪圖分割數量
    "START_EPISODE": 0,             // 續訓時會自動更新
    "BEST_STEPS": 0,                // 續訓時會自動更新
    "BEST_REWARD": null,            // 續訓時會自動更新 (null 會被轉為 -inf)
    "BEST_BALANCE": null,           // 續訓時會自動更新 (null 會被轉為 -inf)
    "SEQUENCE_LENGTH": 30,
    "MEMORY_LIMIT_GB": 12.0,
    "USE_LINE_BOT": false
    // ... 其他 setting.json 中可能存在的參數 (如果實現了更全面的保存)
}
```
**注意**：首次運行時，`START_EPISODE`, `BEST_REWARD`, `BEST_BALANCE` 等應設為初始值。腳本會在中斷時更新這些值。

### 4. 開始訓練

運行主訓練腳本：

```bash
python train_g.py
```

訓練過程中，相關信息會打印到終端，圖表會保存到 `training_plots/`，模型會保存到 `MODEL_PATH` 。

### 5. 斷點續訓

如果訓練中斷（例如手動 Ctrl+C），腳本會嘗試保存當前進度到 `setting.json` 和模型文件。下次直接運行 `python train_g.py`，腳本會從 `setting.json` 中讀取 `START_EPISODE` 等參數，並加載 `LOAD_MODEL_PATH` 指定的模型（通常是 `MODEL_PATH` 下的最新模型或特定最佳模型），從而繼續訓練。

## 主要組件說明

### `trading_env.py`

*   實現了自定義的 Gym 交易環境 `CryptoTradingEnv`。
*   負責數據加載、狀態生成（包含原始市場數據、技術指標和賬戶狀態）、動作執行（買入、賣出、持有）、獎勵計算、以及回合結束判斷。
*   狀態表示和獎勵函數經過精心設計以引導模型學習。

### `dqn_gru_model.py`

*   定義了 DDQN 神經網絡的模型架構。
*   採用了 Inception 模塊和 GRU 層的組合，以有效提取時間序列數據中的特徵。

### `train_g.py`

*   主訓練邏輯腳本。
*   包含了參數加載、環境和模型初始化、Scaler 的擬合與加載、DDQN 算法的實現（包括經驗回放、混合抽樣、目標Q值計算、模型訓練）、學習率和探索率調度、模型保存與加載、進度報告、結果評估和繪圖等功能。

## 未來可能的改進方向

*   **更精細的風險管理**：在獎勵函數中引入更直接的風險指標，如波動率懲罰、夏普比率獎勵等。
*   **倉位管理優化**：讓模型學習動態調整交易倉位大小，而不是固定的百分比。
*   **多時間週期分析**：結合不同時間週期（如 4小時、日線、周線）的數據作為輸入。
*   **更先進的 DRL 算法**：嘗試如 PPO、A2C/A3C 等其他算法。
*   **集成實盤交易接口**：將訓練好的模型部署到真實的交易平台。

## 免責聲明

本項目僅為學術研究和技術演示目的，**不構成任何投資建議**。加密貨幣交易具有高風險性，請謹慎對待。任何基於本項目的實際交易行為所產生的盈虧，均由使用者自行承擔。