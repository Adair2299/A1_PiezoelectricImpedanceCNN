# 修改后的预测，正确的，
# 使用等距索引绘图，横坐标标签为原始列编号，看看到底哪个数据预测的不准
# 1-500频谱，60天的， 四个周期全部
# 没用6号钢板，只用58组

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


# 加载数据
Imp1_500orgl = np.load('../Latest1/Imp1_500.npy')  # (2496,116)
mass_loss_rateorgl = np.load('../Latest1/mass_loss_rate.npy') # (116,1)


# 拆分数据
nsplit = 16
Imp_new = Imp1_500orgl.reshape(int(len(Imp1_500orgl)/nsplit),-1)
mass_new = np.tile(mass_loss_rateorgl, (nsplit, 1))

# 局部方差自适应
from FunA_Adapt_Smoothing import adaptive_smooth
Imp_new = adaptive_smooth(Imp_new, base_sigma=20, window=50, alpha=25)

# --------打乱--------
np.random.seed(42)  # 保证可复现
original_indices = np.arange(len(mass_new))  # 记录原始索引（转置前的列号=mass_new行号）
shuffled_indices = np.random.permutation(len(mass_new))

# 使用相同索引打乱
mass_loss_rate = mass_new[shuffled_indices]
Imp_new = Imp_new.T
Imp1_500 = Imp_new[shuffled_indices]
original_indices_shuffled = original_indices[shuffled_indices]  # 对原始索引也进行相同打乱

##########################################################
# --------------------- 第1步：整理数据 -------------------
sequence_length = Imp1_500.shape[1]  # 用2496个点预测mass_loss

X = np.array(Imp1_500)  # 阻抗数据
y = np.array(mass_loss_rate)  # 质量损失

# **修改点：仅标准化 X，y 直接保持物理值**
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std  # 标准化 X
# y 保持原始尺度 (y 的单位是质量损失，不能归一化)

# **严格按时间顺序划分数据集**
split_time = int(0.7 * len(X))  # 70% 训练
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
train_indices = original_indices_shuffled[:split_time]
remaining_indices = original_indices_shuffled[split_time:]

X_val, X_test = X_remaining[:int(len(X_remaining) // 1.1)], X_remaining[int(len(X_remaining) // 1.1):] # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:int(len(y_remaining) // 1.1)], y_remaining[int(len(y_remaining) // 1.1):]
val_indices = remaining_indices[:int(len(X_remaining) // 1.1)]
test_indices = remaining_indices[int(len(X_remaining) // 1.1):]

# --------------------- 第2步：构建容易过拟合的模型 --------------------
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

    # Initial Conv
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    # x = Conv1D(64, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)

    # Residual Blocks
    x = residual_block(x, 128, 3)
    # x = residual_block(x, 64, 3)

    # Attention Mechanisms
    x = ChannelAttention(x)
    x = TemporalAttention(x)

    # Decoder-like Flatten + FC
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    return model



model = build_advanced_model(sequence_length)

# 1. 定义优化器和回调
optimizer = Adam(learning_rate=0.001)  # 初始学习率
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.9, patience=64, min_lr=7.5e-4, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=96, restore_best_weights=True, verbose=1
)
callbacks = [lr_scheduler, early_stop]  # 组合回调

# 2. 编译模型（替换原有编译代码）
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # 增加MAE监控
model.summary()


# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=500, # 迭代次数
    batch_size=128, # 一次用几个训练
    validation_data=(X_val, y_val),
    callbacks = callbacks,
    verbose=1 # 进度条
)







# --------------------- 第4步：测试集评估 --------------------

# 模型预测
y_test_pred = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

# 计算误差
train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae = mean_absolute_error(y_test.squeeze(), y_test_pred)
test_absolute_errors = np.abs(y_test.squeeze() - y_test_pred.squeeze())
epsilon = 1e-10
test_relative_errors = np.abs((y_test.squeeze() - y_test_pred.squeeze()) / (y_test.squeeze() + epsilon)) * 100


# --------------------- 打印性能报告 --------------------
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




# --------------------- 绘图设置 --------------------
# 1. 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10



# **绘制训练过程**
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=0.5)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=0.5)
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
# plt.ylim(0,0.006)
plt.pause(0.1)



# --------------------- 子图 --------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

# 使用等距索引绘图，横坐标标签为原始列编号
# idx_test 为原始列编号（对应 Imp_new 转置前的列号 / mass_new 的行号）
idx_test = test_indices
plot_idx = np.arange(len(y_test))  # 等距索引

ax1.plot(plot_idx, y_test.squeeze()*100, 'bo-', label='True Value', markersize=4)
ax1.plot(plot_idx, y_test_pred*100, 'rx--', label='Predicted Value', markersize=4)
ax1.set_title('Test Set Prediction Comparison', fontweight='bold')
ax1.set_ylabel('Mass Loss Rate (%)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 25)
ax1.set_xlim(-0.5, len(y_test)-0.5)
ax1.set_xticks(plot_idx)
ax1.set_xticklabels(idx_test)  # 显示原始列编号

ax2.bar(plot_idx, test_absolute_errors*100, color='orange', alpha=0.7, label='Absolute Error')
ax2.set_title('Absolute Errors', fontweight='bold')
# ax2.set_xlabel('Original Sample Column')
ax2.set_ylabel('Absolute Error (%)')
ax2.legend()
ax2.grid(True)
ax2.set_xticks(plot_idx)
ax2.set_xticklabels(idx_test)



# ------------------ 生成编号到 PxxTnX 的映射 ------------------
# 板号（跳过 P6）
plates = [f"P{i}" for i in range(1, 31) if i != 6]
# 周期
cycles = [f"T{t}" for t in range(1, 5)]
# 信号组（A到P，共16个）
signal_groups = [chr(ord('A') + i) for i in range(16)]

# 生成映射表：index -> "PxxTnX"
mapping = []
for sg in signal_groups:
    for cyc in cycles:
        for plate in plates:
            mapping.append(f"{plate}{cyc}{sg}")

# 检查映射长度
assert len(mapping) == 1856, f"映射长度不对: {len(mapping)}"

# -------------- 在 ax2 下方加第二行标签 -----------------
# 从 mapping 中取出对应 idx_test 的标签
converted_labels = [mapping[i] for i in idx_test]

# 创建第二行标签
for x, label in zip(plot_idx, converted_labels):
    ax2.text(x, -max(test_absolute_errors)*100*0.1,  # 位置在x轴下方一点
             label, rotation=45, ha='right', va='top', rotation_mode='anchor', fontsize=9)




plt.tight_layout()
plt.show()
