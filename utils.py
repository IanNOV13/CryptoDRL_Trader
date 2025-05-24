import os
import sys
import json
import tensorflow as tf
import requests # for Discord
# from linebot.v3.messaging import ApiClient, Configuration, MessagingApi, PushMessageRequest, TextMessage # if using LINE

# --- 通知相關 ---
def notify_discord_webhook(message_or_filepath, send_type="text", webhook_url=None):
    """通過 Discord Webhook 發送消息或圖片。"""
    if webhook_url is None:
        webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("[通知] Discord Webhook URL 未設置，跳過通知。")
        return

    try:
        if send_type == "text":
            headers = {"Content-Type": "application/json"}
            data = {"content": message_or_filepath, "username": "訓練回報"}
            res = requests.post(webhook_url, headers=headers, json=data)
            # res.raise_for_status() # 如果需要對非2xx狀態碼拋出異常
        elif send_type == "image":
            if not os.path.exists(message_or_filepath):
                print(f"[錯誤] 圖片文件未找到: {message_or_filepath}")
                return
            with open(message_or_filepath, 'rb') as f:
                files = {'file': (message_or_filepath, f)}
                data = {"content": "", "username": "訓練回報"}
                res = requests.post(webhook_url, data=data, files=files)
            # res.raise_for_status()
        # print(f"Discord 通知發送: {res.status_code}") # 可選的調試信息
    except requests.exceptions.RequestException as e:
        print(f"[錯誤] 發送 Discord 通知失敗: {e}")
    except Exception as e:
        print(f"[錯誤] Discord 通知過程中發生未知錯誤: {e}")

# 由於line的次數限制，移除此傳送方式
# def notify_line_bot(message, user_id, access_token):
#     ... (實現 LINE Bot 推送邏輯) ...

# --- 終端相關 ---
def clear_lines(n=1):
    """清除終端機中光標前的 n 行。"""
    for _ in range(n):
        sys.stdout.write("\033[F")  # 將光標向上移動一行
        sys.stdout.write("\033[K")  # 清除從光標到行尾的內容
        # 如果需要清除整行（而不僅僅是光標之後），可以先 sys.stdout.write("\r")
    sys.stdout.flush()


# --- 系統與環境相關 ---
def setup_gpu(memory_growth=True):
    """配置 TensorFlow GPU 選項。"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU 內存增長已啟用。")
            # 可以添加其他 GPU 配置，如限制顯存使用
            # tf.config.experimental.set_virtual_device_configuration(...)
        except RuntimeError as e:
            print(f"[錯誤] GPU 初始化失敗: {e}")
    else:
        print("未檢測到 GPU，將使用 CPU。")

def set_mixed_precision_policy(policy_name='mixed_float16'):
    """設置混合精度策略。"""
    if policy_name:
        try:
            policy = tf.keras.mixed_precision.Policy(policy_name)
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"混合精度策略已設置為: {tf.keras.mixed_precision.global_policy().name}")
        except Exception as e:
            print(f"[錯誤] 設置混合精度策略失敗: {e}")