import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, GlobalAveragePooling1D, Multiply, Permute
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from matplotlib import rcParams
matplotlib.use('Qt5Agg')  # 解决plt警告

# 加载数据
Imp1_500 = np.load('../Latest1/Imp1_500.npy')  # 原始形状 (2496, 116)
mass_loss_rate = np.load('../Latest1/mass_loss_rate.npy')  # (116, 1)


# 局部方差自适应平滑（如果你有此函数）
from FunA_Adapt_Smoothing import adaptive_smooth
Imp1_500 = adaptive_smooth(Imp1_500, base_sigma=20, window=50, alpha=25)

# 打乱样本
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))  # 116
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]

# 准备训练和测试数据
sequence_length = Imp1_500.shape[1]  # 2496

X = Imp1_500.T
y = mass_loss_rate

# 标准化 X
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std

# 严格按样本顺序拆分数据集
split_time = int(0.7 * len(X))  # 81 训练样本
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining)//2], X_remaining[len(X_remaining)//2:]
y_val, y_test = y_remaining[:len(y_remaining)//2], y_remaining[len(y_remaining)//2:]

# 转换为3D输入模型需要的格式 [样本数, 时间步长, 特征数]
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)


# 定义注意力模块和残差块

def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])

def TemporalAttention(input_tensor):
    permuted = Permute((2, 1))(input_tensor)  # [B, C, T]
    dense = Dense(input_tensor.shape[1], activation='softmax')(permuted)
    attention = Permute((2, 1))(dense)  # [B, T, C]
    return Multiply()([input_tensor, attention])

def residual_block(x, filters, kernel_size, pooling=True):
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)  # 1x1 conv调整通道
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Add()([shortcut, x])
    x = LayerNormalization()(x)
    if pooling:
        x = MaxPooling1D(pool_size=2)(x)
    return x


def build_advanced_model(input_length):
    inputs = Input(shape=(input_length, 1))
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)
    x = residual_block(x, 128, 3)
    x = ChannelAttention(x)
    x = TemporalAttention(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model


model = build_advanced_model(sequence_length)

optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
callbacks = [lr_scheduler, early_stop]

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=12,
    validation_data=(X_val, y_val),
    # callbacks=callbacks,
    verbose=1
)

# 评估
y_test_pred = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae = mean_absolute_error(y_test.squeeze(), y_test_pred)
test_absolute_errors = np.abs(y_test.squeeze() - y_test_pred)
epsilon = 1e-10
test_relative_errors = np.abs((y_test.squeeze() - y_test_pred) / (y_test.squeeze() + epsilon)) * 100

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

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10

plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=0.5)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=0.5)
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.pause(0.1)

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
