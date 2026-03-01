# 04月27日可行,06月07日可行，基于004普通神经网络的改进预测massloss
# 1-500频谱，60天的， 四个周期全部
# 没用6号钢板，只用58组
# 残差块包裹没注意力机制
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, \
    GlobalAveragePooling1D, Multiply, Permute, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams

matplotlib.use('Qt5Agg')  # 解决plt警告

# 加载数据
Imp1_500 = np.load('../../Latest1/Imp1_500.npy')  # (2496,116)
mass_loss_rate = np.load('../../Latest1/mass_loss_rate.npy')

# 局部方差自适应
from FunA_Adapt_Smoothing import adaptive_smooth

Imp1_500 = adaptive_smooth(Imp1_500, base_sigma=20, window=50, alpha=25)

# 打乱数据
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]

# --------------------- 数据整理 -------------------
sequence_length = Imp1_500.shape[1]
X = np.array(Imp1_500)
y = np.array(mass_loss_rate)

X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std

split_time = int(0.7 * len(X))
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:]
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)


# --------------------- 模型构建 -------------------
def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])


def TemporalAttention(input_tensor):
    permuted = Permute((2, 1))(input_tensor)
    dense = Dense(input_tensor.shape[1], activation='softmax')(permuted)
    attention = Permute((2, 1))(dense)
    return Multiply()([input_tensor, attention])


def attention_residual_block(x, filters, kernel_size, pooling=True):
    shortcut = x

    # 主路径
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)

    # 嵌入通道注意力
    x = ChannelAttention(x)

    # 嵌入时序注意力
    x = TemporalAttention(x)

    x = Dropout(0.2)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)

    # 维度匹配
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    x = Add()([shortcut, x])
    x = LayerNormalization()(x)

    if pooling:
        x = MaxPooling1D(pool_size=2)(x)
    return x


def build_advanced_model(input_length):
    inputs = Input(shape=(input_length, 1))

    # 初始卷积
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)

    # 残差块（内部包含注意力）
    x = attention_residual_block(x, 128, 3)

    # 输出部分
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    return model


model = build_advanced_model(sequence_length)

# --------------------- 训练配置 -------------------
optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=500, restore_best_weights=True, verbose=1
)
callbacks = [lr_scheduler, early_stop]

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# --------------------- 训练模型 -------------------
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=12,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=callbacks
)

# --------------------- 结果可视化 -------------------
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10

plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.ylim(0, 0.006)
plt.show()

# 模型预测
y_test_pred = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

# 计算误差
train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae = mean_absolute_error(y_test.squeeze(), y_test_pred)
test_absolute_errors = np.abs(y_test.squeeze() - y_test_pred.squeeze())
epsilon = 1e-10
test_relative_errors = np.abs((y_test.squeeze() - y_test_pred.squeeze()) / (y_test.squeeze() + epsilon)) * 100

# 打印报告
print(f'''
=== 性能报告 ===
训练样本数：{len(y_train)}，测试样本数：{len(y_test)}
训练集 MAE: {train_mae:.4f}
测试集 MAE: {test_mae:.4f}
误差增加: {((test_mae - train_mae) / train_mae) * 100:.1f}%

--- 测试集详细误差分析 ---
绝对误差范围: [{np.min(test_absolute_errors):.4f}, {np.max(test_absolute_errors):.4f}]
平均绝对误差: {np.mean(test_absolute_errors):.4f}

相对误差范围: [{np.min(test_relative_errors):.2f}%, {np.max(test_relative_errors):.2f}%]
平均相对误差: {np.mean(test_relative_errors):.2f}%
''')

# 绘制结果对比
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
sample_numbers = np.arange(1, len(y_test) + 1)

ax1.plot(sample_numbers, y_test * 100, 'bo-', label='True Value', markersize=4)
ax1.plot(sample_numbers, y_test_pred * 100, 'rx--', label='Predicted Value', markersize=4)
ax1.set_title('Test Set Prediction Comparison', fontweight='bold')
ax1.set_ylabel('Mass Loss Rate (%)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 25)
ax1.set_xlim(0.5, len(y_test) + 0.5)

ax2.bar(sample_numbers, test_absolute_errors * 100, color='orange', alpha=0.7, label='Absolute Error')
ax2.set_title('Absolute Errors', fontweight='bold')
ax2.set_xlabel('Sample Number')
ax2.set_ylabel('Absolute Error (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()