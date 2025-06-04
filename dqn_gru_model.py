import tensorflow as tf
from tensorflow.keras import Model# type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, LeakyReLU, Input , Concatenate, Add, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.layers import BatchNormalization # type: ignore
import numpy as np
# --- 你的 Inception 模塊實現 ---
def inception_module(input_tensor):
    # 使用 256 個濾波器作為 bottleneck 和主要卷積層的濾波器數量
    # MaxPooling 分支的 bottleneck 使用 128 個濾波器
    num_filters_conv = 256
    num_filters_bottleneck = 256
    num_filters_mp_bottleneck = 128

    bottleneck = Conv1D(filters=num_filters_bottleneck, kernel_size=1, padding='same', activation=None, use_bias=False)(input_tensor)
    conv3 = Conv1D(filters=num_filters_conv, kernel_size=3, padding='same', activation=None, use_bias=False)(bottleneck)
    conv7 = Conv1D(filters=num_filters_conv, kernel_size=7, padding='same', activation=None, use_bias=False)(bottleneck)
    conv14 = Conv1D(filters=num_filters_conv, kernel_size=14, padding='same', activation=None, use_bias=False)(bottleneck)
    conv28 = Conv1D(filters=num_filters_conv, kernel_size=28, padding='same', activation=None, use_bias=False)(bottleneck)

    mp = MaxPooling1D(pool_size=8, strides=1, padding='same')(input_tensor) # 注意 pool_size=8
    mpbottleneck = Conv1D(filters=num_filters_mp_bottleneck, kernel_size=1, padding='same', activation=None, use_bias=False)(mp)

    x = Concatenate(axis=-1)([conv3, conv7, conv14, conv28, mpbottleneck])
    x = BatchNormalization()(x)
    # 使用 LeakyReLU 替換 Activation('LeakyReLU') 以便指定 alpha (如果需要)
    x = LeakyReLU(alpha=0.1)(x) # 或者保持 Activation('LeakyReLU') 如果你的 TF 版本默認行為符合預期
    return x

# --- 你的 Shortcut (Residual) 層實現 ---
def shortcut_layer(input_tensor1, input_tensor2):
    # input_tensor1 是 shortcut 的來源, input_tensor2 是主路徑的輸出
    shortcut = Conv1D(filters=input_tensor2.shape[-1], kernel_size=1, padding='same', activation=None, use_bias=False)(input_tensor1)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, input_tensor2])
    # 使用 LeakyReLU 替換 Activation('LeakyReLU')
    x = LeakyReLU(alpha=0.1)(x) # 或者保持 Activation('LeakyReLU')
    return x

# --- 新的模型構建函數 (使用你提供的模塊) ---
def build_inception_gru_dqn_model(input_shape, action_size):
    """
    使用你提供的 Inception+Residual+GRU 架構作為特徵提取器，
    並連接到原有的 DQN Dense 頭部來輸出 Q 值。
    假設 input_shape = (n_time_steps, n_features)
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2 and input_shape[0] > 1):
         # 強調 n_time_steps 必須大於 1
         raise ValueError("input_shape 必須是 (n_time_steps, n_features) 且 n_time_steps > 1")

    #n_time_steps, n_features = input_shape
    input_layer = Input(shape=input_shape, name="input")

    # === 特徵提取: 基於第二個模型的架構 ===
    x = input_layer
    input_residual = input_layer # 用於 shortcut 連接

    num_inception_blocks = 6 # 來自你的描述

    for i in range(num_inception_blocks):
        x = inception_module(x) # 直接調用你定義的 inception_module

        # --- 處理殘差連接 (基於第二個模型的描述 i%3 == 2) ---
        if i % 3 == 2:
            x = shortcut_layer(input_residual, x) # 調用你定義的 shortcut_layer
            input_residual = x # 更新下一個 shortcut 的輸入

    # --- GRU 層 (來自模型 2) ---
    # return_sequences=True 保持序列輸出給下一層 GRU
    x = GRU(512, return_sequences=True, dropout=0.1, name="gru_1")(x)
    # 第二層 GRU - 決定如何獲取最終特徵向量
    # 選擇 1: 使用 GlobalAveragePooling1D (同模型 2)
    x = GRU(512, return_sequences=True, dropout=0.1, name="gru_2")(x) # 保持 return_sequences=True
    feature_vector = GlobalAveragePooling1D(name="gap_gru")(x)

    # 選擇 2: 使用最後一個時間步的輸出
    # x = GRU(512, return_sequences=False, dropout=0.1, name="gru_2")(x) # 設置 return_sequences=False
    # feature_vector = x

    # === DQN 頭部: 使用你第一個模型的 Dense 層結構 ===
    dqn_head = Dense(64, name="dqn_dense_1")(feature_vector)
    dqn_head = LeakyReLU(alpha=0.1, name="dqn_relu_1")(dqn_head)

    # 最終輸出 Q 值
    output_layer = Dense(action_size, dtype='float32', activation='linear', name="q_values")(dqn_head)

    # === 構建和編譯模型 ===
    model = Model(inputs=input_layer, outputs=output_layer)
    # 使用與第一個模型相同的優化器和損失函數
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-7))

    return model

if __name__ == "__main__":
    model = build_inception_gru_dqn_model(input_shape=(24, 11), action_size=3)
    model.summary()
