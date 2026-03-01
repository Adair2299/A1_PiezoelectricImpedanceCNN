import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, GlobalAveragePooling1D, Multiply, Permute, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
matplotlib.use('Qt5Agg')  # 解决plt警告

# ----------------- 修正后的数据加载部分 -----------------
Imp1_500 = np.load('../Latest1/Imp1_500.npy')  # 实际形状 (2496,116)
mass_loss_rate = np.load('../Latest1/mass_loss_rate.npy')  # 形状应为 (116,)

# 转置数据使样本在第一个维度 (116,2496)
Imp1_500 = Imp1_500.T  # 现在形状 (116,2496)

# 保存打乱前的原始数据
Imp1_500_orig = Imp1_500.copy()

# 局部方差自适应
from FunA_Adapt_Smoothing import adaptive_smooth
Imp1_500 = adaptive_smooth(Imp1_500, base_sigma=20, window=50, alpha=25)

# 打乱数据 (保持样本和标签对应)
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]

# ----------------- 修正后的数据整理 -----------------
sequence_length = Imp1_500.shape[1]  # 2496 (频率点数)

X = np.array(Imp1_500)  # (116,2496)
y = np.array(mass_loss_rate)  # (116,)

# 标准化 X (按样本标准化)
X_mean, X_std = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
X = (X - X_mean) / X_std

# 划分数据集 (样本维度是116)
split_time = int(0.7 * len(X))  # 70% 训练
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining)//2], X_remaining[len(X_remaining)//2:]
y_val, y_test = y_remaining[:len(y_remaining)//2], y_remaining[len(y_remaining)//2:]

# 调整形状为 (样本数, 序列长度, 1)
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])


def TemporalAttention(input_tensor):
    permuted = Permute((2, 1))(input_tensor)  # [B, C, T]
    dense = Dense(input_tensor.shape[1], activation='softmax', name='time_attention_softmax')(permuted)  # 给层命名，方便提取
    attention = Permute((2, 1))(dense)  # [B, T, C]
    output = Multiply()([input_tensor, attention])
    return output, dense  # 返回乘积输出和注意力权重


def residual_block(x, filters, kernel_size, pooling=True):
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Add()([shortcut, x])
    x = LayerNormalization()(x)
    if pooling:
        x = MaxPooling1D(pool_size=2)(x)
    return x


def build_advanced_model_with_attention_weights(input_length):
    inputs = Input(shape=(input_length, 1))

    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)

    x = residual_block(x, 128, 3)

    x = ChannelAttention(x)

    x, time_attention_weights = TemporalAttention(x)  # 改为接收注意力权重张量

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    # 新增一个模型专门输出时间注意力权重
    attention_model = Model(inputs, time_attention_weights)

    return model, attention_model


model, attention_model = build_advanced_model_with_attention_weights(sequence_length)

optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
callbacks = [lr_scheduler, early_stop]

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# 训练代码保持不变
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=12,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=callbacks
)

# 评估代码保持不变...

# --- 新增部分：可视化时间注意力机制权重 ---

# --- 新增部分：可视化时间注意力机制权重 ---

# 定义 sample_idx ，这里是你想看打乱前的第几个样本，自己设定
sample_idx = 80  # 比如第0个样本，可以修改

# 由于你打乱了数据，现在要找到打乱前的原始数据对应索引
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))
inverse_idx = np.where(shuffled_indices == sample_idx)[0][0]

# 获取原始信号（未标准化的）
original_signal = Imp1_500_orig[inverse_idx]  # 使用保存的原始数据

# 取对应样本的输入（标准化后）
x_sample = X[inverse_idx]  # X是标准化后全部数据，形状(2496, sequence_length)
x_sample = x_sample.reshape(1, sequence_length, 1)  # 批量维度

# 预测该样本的时间注意力权重
time_attention_weights = attention_model.predict(x_sample)  # (1, channel, time_steps)

# 获取实际的注意力权重长度
attention_length = time_attention_weights.shape[-1]

# 对通道维度做平均，得到每个时间步的总体注意力权重：
attention_per_time = np.mean(time_attention_weights, axis=1).squeeze()  # (T,)

# 创建频率轴
original_freqs = np.linspace(1, 500, sequence_length)  # 原始信号频率轴
attention_freqs = np.linspace(1, 500, attention_length)  # 注意力权重频率轴

plt.plot(original_freqs,Imp1_500_orig[:,82])
plt.show()

# 创建包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 上子图：原始信号
ax1.plot(original_freqs, original_signal, label='Original Signal', color='blue')
ax1.set_ylabel('Conductance (mS)')
ax1.set_title(f'Sample {sample_idx} - Original Signal and Attention Weights')
ax1.grid(True)
ax1.legend()

# 下子图：注意力权重
ax2.plot(attention_freqs, attention_per_time, label='Time Attention Weight', color='red')
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('Attention Weight')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()