# 基於深度強化學習的加密貨幣交易智能體

本項目旨在使用深度強化學習 (DRL) 技術，特別是雙重深度Q網絡 (DDQN) 算法，訓練一個能夠在加密貨幣市場（以 BTCUSDT 為例）進行自動化交易的智能體。通過精心設計的狀態表示、動態獎勵函數以及先進的學習策略，致力於實現超越傳統買入並持有策略的交易表現。

## 項目核心特性

*   **先進的DRL算法**：採用 DDQN (Double Deep Q-Network) 算法，緩解Q值過高估計問題，提升學習穩定性和策略質量。
*   **精細化狀態表示**：結合市場原始K線數據 (OHLCV)、多種技術分析指標，以及實時計算的賬戶狀態（如持倉價值佔總資產比例、總資金相對初始本金比例、未交易時長、持倉相對成本的符號平方根）構建強化學習的狀態空間，為模型提供豐富決策依據。
*   **動態與多目標獎勵函數**：獎勵函數綜合考慮資產變化、單筆交易盈虧、風險回撤控制、交易活躍度及無效操作懲罰。特別引入與持倉時長相關的動態交易獎勵尺度，以抑制高頻刷分行為並鼓勵耐心等待後的精準操作。
*   **靈活的超參數配置與進度管理**：通過 `setting.json` 文件集中管理所有關鍵訓練參數（如學習率、探索率、折扣因子、網絡結構參數等），方便進行實驗調優。同時，訓練進度（如當前回合、最佳獎勵、最佳餘額等）也會在程序中斷時自動保存回此文件，支持斷點續訓。
*   **穩健的數據預處理**：集成 `StandardScaler` (或可選的 `RobustScaler`) 進行狀態特徵歸一化，並包含自動化的 Scaler 對象加載、擬合（基於訓練集）與保存機制。
*   **全面的訓練監控與可視化**：
    *   實時在終端更新並打印詳細的訓練進度，包括當前回合、步數、探索率、學習率、各項獎勵/懲罰分解、交易統計（高/低賣次數及比例）、系統資源佔用 (RAM) 以及賬戶餘額詳情。
    *   可選通過 Discord (或 LINE Bot) 推送每回合的訓練總結報告和階段性的模型評估結果。
    *   定期（如每25回合或達到新的最佳記錄時）繪製包含價格和買賣點的交易行為圖表，並分段保存圖像文件。
*   **可靠的斷點續訓與模型保存**：
    *   能夠從上次訓練中斷的地方（由 `setting.json`記錄）無縫繼續。
    *   通過捕獲 `SIGINT` (Ctrl+C) 和 `SIGTERM` 系統中斷信號，實現程序意外退出前的安全模型保存和進度更新。
    *   根據訓練過程中的最佳表現（如最高餘額、最高獎勵、最長有效步數），分別保存對應的模型版本。
*   **優化的經驗回放策略**：採用多經驗池混合抽樣機制，按比例從主經驗池、歷史最佳餘額回合經驗池和歷史最佳獎勵回合經驗池中抽取樣本進行訓練，以強化對成功經驗模式的學習。
*   **先進的學習率與探索率調度**：
    *   學習率採用帶有熱重啟的餘弦退火 (Cosine Annealing with Warm Restarts) 調度策略，基於全局訓練步數動態調整。
    *   探索率 (Epsilon) 採用週期性的餘弦退火調度，基於回合數動態調整，旨在平衡探索與利用，並在訓練後期提供重新探索的機會。
*   **模塊化與結構化代碼**：項目代碼被清晰地劃分為環境定義 (`trading_env.py`)、神經網絡模型構建 (`dqn_gru_model.py`)、智能體核心邏輯 (`DQNAgent.py`)、主訓練流程 (`main.py` 或 `train_g.py`)、數據處理 (`data_handler.py`, `create_technical_data.py`)、配置加載 (`config_loader.py`)、評估 (`evaluator.py`)、繪圖 (`plotter.py`) 及通用工具 (`utils.py`) 等模塊，提高了代碼的可讀性、可維護性和可擴展性。

## 環境要求

*   Python 3.7+
*   TensorFlow 2.x (推薦 TensorFlow >= 2.6 以獲得更好的 `tf.function` 和混合精度支持)
*   Gymnasium (或 Gym 0.21+，建議使用 Gymnasium 以獲得持續支持)
*   NumPy
*   Pandas
*   Pandas-TA (用於快速計算技術指標)
*   Matplotlib (用於繪製圖表)
*   Scikit-learn (用於 `StandardScaler` 或 `RobustScaler`)
*   python-dotenv (用於從 `.env` 文件加載環境變量)
*   requests (用於與 Discord Webhook 交互)
*   joblib (用於高效保存/加載 Python 對象，如 Scikit-learn 的 Scaler)
*   (可選) `line-bot-sdk>=3.0.0` (如果啟用 LINE Bot 通知功能)
*   (可選) `keyboard` (如果需要手動鍵盤暫停功能，注意此庫在某些系統可能需要特定權限)

## 項目結構 (示例)

```
.
├── full_csv/                  # 存放原始市場數據 CSV 文件 (例如: BTCUSDT_1d_full.csv)
├── technical/                 # 存放計算了技術指標的數據文件 (例如: BTCUSDT_1d_technical.csv)
├── training_plots/            # 存放訓練過程中生成的交易行為圖表 (按回合分子目錄)
├── BTC_1d_model               # 具體某次訓練的模型保存目錄
│   └── scaler.joblib          # 對應的 Scaler 對象
├── config_loader.py           # 配置加載與管理模塊
├── create_technical_data.py   # 技術指標生成腳本
├── data_handler.py            # 數據準備與環境創建封裝模塊
├── dqn_gru_model.py           # 神經網絡模型架構定義 (Inception+GRU DDQN)
├── DQNAgent.py                # DDQN 智能體核心邏輯實現
├── evaluator.py               # 模型評估函數模塊
├── main.py                    # 主訓練與執行腳本 (或者你的 train_g.py)
├── plotter.py                 # 交易行為繪圖模塊
├── trading_env.py             # 自定義的 Gym 交易環境實現
├── utils.py                   # 通用工具函數模塊
├── setting.json               # 項目配置文件，存儲訓練參數和進度
├── .env                       # 存儲 API Keys, Webhook URLs 等敏感信息
├── requirements.txt           # Python 依賴列表
└── README.md                  # 本項目說明文件
```

## 使用說明

### 1. 環境設置與依賴安裝
   - 克隆本倉庫。
   - 創建並激活 Python 虛擬環境。
   - 運行 `pip install -r requirements.txt` 安裝所需依賴。

### 2. 準備數據
   - 將你的原始市場數據 (OHLCV格式，包含 `timestamp`, `open`, `high`, `low`, `close`, `volume` 列) 保存為 CSV 文件，放置於 `full_csv/` 目錄下，例如 `full_csv/BTCUSDT_1d_full.csv`。
   - 運行 `create_technical_data.py` 腳本為原始數據計算並添加技術指標。腳本會讀取 `full_csv/` 中的文件，並將處理後的數據保存到 `technical/` 目錄。
     ```bash
     python create_technical_data.py
     ```
   - 確保 `data_handler.py` 或環境直接引用的數據文件路徑指向 `technical/` 目錄中已生成指標的文件。

### 3. 配置環境變量
   - 複製 `.env.example` (如果提供) 為 `.env`，或者手動創建 `.env` 文件。
   - 在 `.env` 文件中填寫必要的 API 密鑰或 Webhook URL，例如：
     ```env
     DISCORD_WEBHOOK="你的Discord_Webhook_URL_用於接收訓練通知"
     # LINE_CHANNEL_ACCESS_TOKEN="你的LINE_Channel_Access_Token" # 因為LINE的傳送次數限制已取消這個選項
     # USER_ID="你的LINE_User_ID" # 因為LINE的傳送次數限制已取消這個選項
     ```

### 4. 配置訓練參數 (`setting.json`)
   - 首次運行前，請仔細檢查和配置 `setting.json` 文件。關鍵參數包括：
     - `LOAD_MODEL_PATH`: 如果是全新訓練，可以設置為 `MODEL_PATH` 或一個不存在的路徑。如果是續訓，指向要加載的模型目錄。
     - `MODEL_PATH`: 模型及相關文件（如Scaler）的保存基礎目錄。
     - `EPISODES`: 總訓練回合數。
     - `BATCH_SIZE`, `GAMMA`, `SEQUENCE_LENGTH`: DQN 核心參數。
     - Epsilon 相關參數 (`INITIAL_EPSILON`, `MIN_EPSILON_CYCLE`, `EPSILON_CYCLE_LENGTH`, `EPSILON_MIN`)。
     - 學習率相關參數 (`INITIAL_LR`, `LR_DROP_RATE`, `EPOCHS_DROP_RL`, `ALPHA_LR`)。
     - `START_EPISODE`, `BEST_REWARD`, `BEST_BALANCE`, `global_step_count`: 這些參數通常由腳本自動更新以支持斷點續訓。首次運行時，`START_EPISODE` 應為 0，`BEST_REWARD` 和 `BEST_BALANCE` 應為 `null` (JSON中的null會被腳本正確處理為-np.inf)。
   - 示例 `setting.json` (部分)：
     ```json
     {
         "LOAD_MODEL_PATH": "BTC_1d_model",
         "MODEL_PATH": "BTC_1d_model",
         "EPISODES": 8000,
         "START_EPISODE": 0,
         "BEST_REWARD": null,
         "BEST_BALANCE": null,
         "global_step_count": 0,
         // ... 其他所有在腳本中從 config 讀取的參數 ...
         "SCALER_TYPE": "RobustScaler" // 例如，選擇使用 RobustScaler
     }
     ```

### 5. 開始訓練
   - 運行主訓練腳本 (例如 `main.py` 或 `train_g.py`):
     ```bash
     python main.py
     ```
   - 訓練進度將實時打印在終端。
   - 模型會根據設定的條件（如達到新的最佳記錄、每隔N回合）自動保存。
   - 相關圖表會保存在 `training_plots/` 目錄下。

### 6. 斷點續訓
   - 如果訓練意外中斷 (例如手動按下 Ctrl+C)，腳本會嘗試保存當前的模型和訓練進度 (更新 `setting.json`)。
   - 要繼續訓練，只需再次運行相同的 `python main.py` 命令。腳本會自動從 `setting.json` 中讀取上次的進度（如 `START_EPISODE`）並加載模型，從而繼續訓練。

## 主要組件職責

*   **`main.py` / `train_g.py`**: 組織和驅動整個訓練流程，協調各個模塊。
*   **`config_loader.py`**: 負責從 `setting.json` 加載配置參數。
*   **`data_handler.py`**: 封裝數據準備、環境創建和 Scaler 初始化/擬合的邏輯。
*   **`create_technical_data.py`**: 獨立腳本，用於從原始市場數據計算技術指標。
*   **`trading_env.py`**: 實現符合 Gym(nasium) 接口的加密貨幣交易環境。
*   **`dqn_gru_model.py`**: 定義神經網絡模型架構。
*   **`DQNAgent.py`**: 實現 DDQN 智能體的核心算法，包括動作選擇、經驗存儲、學習更新、目標網絡管理等。
*   **`evaluator.py`**: 包含在測試環境上評估模型性能的函數。
*   **`plotter.py`**: 負責生成交易行為的可視化圖表。
*   **`utils.py`**: 包含項目中多處使用的通用輔助函數（如通知、GPU設置、進度保存到JSON等）。

## 未來可能的改進方向

*   **更精細的風險管理集成**：直接在獎勵函數或狀態中引入如波動率、夏普比率等風險度量。
*   **動態倉位管理**：讓智能體學習根據市場狀況和自身判斷來決定交易的倉位大小。
*   **多資產交易**：將模型擴展到同時交易多種加密貨幣。
*   **集成更先進的DRL算法**：探索如 PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic) 等算法的應用。
*   **超參數自動優化**：使用如 Optuna, Ray Tune 等工具進行超參數搜索。
*   **更全面的回測框架**：集成更專業的回測工具進行嚴格的策略評估。
*   **模擬交易與實盤部署接口**：逐步將訓練好的模型部署到模擬交易環境，並最終考慮接入實盤API。

## 免責聲明

本項目提供的所有代碼和信息僅用於學術研究、技術交流和個人學習目的。**不構成任何形式的投資建議或交易指導。** 加密貨幣市場具有極高的風險性和波動性，任何基於本項目的實際交易行為所導致的盈利或虧損，均由使用者本人獨立承擔全部責任。在進行任何實際投資決策前，請務必進行充分的獨立研究並諮詢專業財務顧問。