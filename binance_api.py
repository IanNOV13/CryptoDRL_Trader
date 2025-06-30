import sqlite3
import os
from binance.client import Client
from datetime import datetime
# ... 其他 import

DB_FILE = "trading_data.db"

def init_db():
    """初始化資料庫和資料表"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                orderId INTEGER,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                isBuyer INTEGER NOT NULL,
                time INTEGER NOT NULL
            )
        """)
        conn.commit()

def sync_trades_to_db(client, symbol: str):
    """同步最新的交易紀錄到本地資料庫"""
    print(f"開始同步 {symbol} 的交易紀錄...")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        
        # 1. 取得本地資料庫中最新的交易 ID
        cursor.execute("SELECT MAX(id) FROM trades WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        last_id = result[0] if result and result[0] is not None else 0

        # 2. 從幣安 API 拉取新交易
        from_id = last_id + 1
        
        while True:
            # 每次最多獲取 1000 筆 (API 上限)
            print(f"正在從 ID {from_id} 開始獲取交易...")
            trades = client.get_my_trades(symbol=symbol, fromId=from_id, limit=1000)
            
            if not trades:
                print("沒有新的交易紀錄。")
                break
            
            # 3. 將新交易寫入資料庫
            for trade in trades:
                cursor.execute("""
                    INSERT OR IGNORE INTO trades (id, symbol, orderId, price, qty, isBuyer, time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade['id'],
                    trade['symbol'],
                    trade['orderId'],
                    float(trade['price']),
                    float(trade['qty']),
                    1 if trade['isBuyer'] else 0,
                    trade['time']
                ))
            
            from_id = trades[-1]['id'] + 1
            conn.commit()
            print(f"已同步 {len(trades)} 筆新交易，最新的 ID 為 {from_id - 1}")

def get_stats_from_db(symbol: str) -> dict:
    """
    從本地資料庫讀取數據，計算 FIFO 平均成本和最後交易時間。
    """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # 一次性查詢所有需要的數據
        cursor.execute(
            "SELECT qty, price, isBuyer, time FROM trades WHERE symbol = ? ORDER BY time ASC",
            (symbol,)
        )
        all_trades_from_db = cursor.fetchall()

    if not all_trades_from_db:
        return {
            'avg_cost': 0,
            'last_trade_time': None,
            'calculated_qty': 0
        }

    # --- FIFO 計算邏輯 (與之前相同) ---
    buy_queue = []
    for qty, price, is_buyer_int, time in all_trades_from_db:
        is_buyer = bool(is_buyer_int)
        if is_buyer:
            buy_queue.append([qty, price])
        else:
            sell_qty = qty
            while sell_qty > 0 and buy_queue:
                buy_qty, buy_price = buy_queue[0]
                if sell_qty < buy_qty:
                    buy_queue[0][0] -= sell_qty
                    sell_qty = 0
                else:
                    sell_qty -= buy_qty
                    buy_queue.pop(0)

    total_qty = sum(q[0] for q in buy_queue)
    total_cost = sum(q[0] * q[1] for q in buy_queue)
    avg_cost_price = total_cost / total_qty if total_qty > 0 else 0

    # --- 取得最後交易時間 ---
    # 因為數據已經按時間排序，最後一筆就是最新的
    last_trade_timestamp_ms = all_trades_from_db[-1][3]
    last_trade_dt = datetime.fromtimestamp(last_trade_timestamp_ms / 1000)
    last_trade_time_str = last_trade_dt.strftime('%Y-%m-%d %H:%M:%S') # 格式可以自訂
    now_dt = datetime.now()

    # 計算時間差並提取天數
    time_difference = now_dt - last_trade_dt
    days_since = time_difference.days

    return {
        'avg_cost': avg_cost_price,
        'last_trade_time': last_trade_time_str,
        'calculated_qty': total_qty,
        'days_since_last_trade': days_since,
    }

def get_account_summary(api_key: str, api_secret: str) -> dict:
    client = Client(api_key, api_secret)
    symbol = 'BTCUSDT'
    base_asset = 'BTC'
    quote_asset = 'USDT'

    try:
        # 步驟 1: 同步交易紀錄到本地DB
        sync_trades_to_db(client, symbol)

        # 步驟 2: 從DB計算統計數據 (成本、最後交易日等)
        db_stats = get_stats_from_db(symbol)
        avg_cost_price = db_stats['avg_cost']
        last_trade_time = db_stats['last_trade_time']
        calculated_qty = db_stats['calculated_qty']
        days_since_last_trade = db_stats['days_since_last_trade']

        # 步驟 3: 查詢即時餘額 (從API)
        btc_balance = client.get_asset_balance(asset=base_asset)['free']
        usdt_balance = client.get_asset_balance(asset=quote_asset)['free']
        
        # 檢查計算數量與實際餘額是否一致
        warning = None
        if abs(calculated_qty - float(btc_balance)) > 1e-6: # 容許浮點數誤差
            warning = (
                f"警告：計算出的 BTC 數量 ({calculated_qty:.8f}) 與 "
                f"API 實際餘額 ({float(btc_balance):.8f}) 不符。成本可能不準確。"
            )
            print(warning)

        summary = {
            'btc_balance': round(float(btc_balance), 8),
            'usdt_balance': round(float(usdt_balance), 2),
            'btc_avg_cost': round(avg_cost_price, 2),
            'last_trade_time': last_trade_time or "無交易紀錄",
            'days_since_last_trade': days_since_last_trade,
            'warning': warning # 將警告訊息也加入回傳字典
        }
        return summary

    except Exception as e:
        print(f"錯誤發生: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    init_db()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    summary = get_account_summary(api_key, api_secret)
    print("\n--- 帳戶總覽 ---")
    # 使用迴圈印出，讓格式更整齊
    for key, value in summary.items():
        print(f"{key}: {value}")