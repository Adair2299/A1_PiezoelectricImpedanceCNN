# 去除质量不好的EMI，注意力嵌套在残差块里
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


# # ------------识别文件1：质量损失------------
# # 指定路径和文件
# folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作")
# file = "3_sasq和质量损失率.xlsx"
#
# # 构建安全路径（自动处理操作系统差异）
# file_path = folder / file
#
# # 检查文件是否存在
# if not file_path.exists():
#     raise FileNotFoundError(f"文件不存在: {file_path}")
#
# # 读取Excel数据
# # 读取 B16:AE16 和 B24:AE24 两行数据
# df1 = pd.read_excel(file_path, header=None, skiprows=15, nrows=1, usecols=range(1, 31))
#
# # 读取第二组数据(B24:AE24)
# df2 = pd.read_excel(file_path, header=None, skiprows=23, nrows=1, usecols=range(1, 31))
# df3 = pd.read_excel(file_path, header=None, skiprows=31, nrows=1, usecols=range(1, 31))
# df4 = pd.read_excel(file_path, header=None, skiprows=39, nrows=1, usecols=range(1, 31))
#
# # 合并两个DataFrame
# result = pd.concat([df1, df2, df3, df4], ignore_index=True)
# result = np.array(result)
# result = np.delete(result, [3,5,17], axis=1) # 4,6,18号钢板数据不要
# mass_loss_rateorgl = result.reshape(-1, 1)
#
#
# # ---------识别文件2：阻抗谱 Piezoelectric Impedance Spectrum---------
#
# # 定义文件路径
#
#
# # 文件路径
# file_path = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term all 1-500 RemoveWrongEMI.xlsx")
#
# # 精确读取 B2:DM2497
# try:
#     df = pd.read_excel(file_path, usecols="B:DE", header=None, skiprows=1, nrows=2496)
#     print(f"成功读取，形状为: {df.shape}")  # 应该是 (2496, 108)
#
# except Exception as e:
#     print(f"读取失败: {e}")
#
# Imp1_500_Orgl = df.values
# print(f"最终数组形状: {Imp1_500_Orgl.shape}")  # 应该是(2496, 108)
#
# Imp1_500orgl = np.array(Imp1_500_Orgl)
#

# 加载数据
Imp1_500orgl = np.load('../../Latest2/Imp1_500_RemoveWrongEMI.npy')  # (2496,116)
mass_loss_rateorgl = np.load('../../Latest2/mass_loss_rate_RemoveWrongEMI.npy') # (116,1)

# 拆分数据
nsplit = 16
Imp_new = Imp1_500orgl.reshape(int(len(Imp1_500orgl)/nsplit),-1)
mass_new = np.tile(mass_loss_rateorgl, (nsplit, 1))

from FunA_Adapt_Smoothing import adaptive_smooth
Imp_new = adaptive_smooth(Imp_new, base_sigma=20, window=50, alpha=25)

# --------打乱--------
np.random.seed(2)
original_indices = np.arange(len(mass_new))
shuffled_indices = np.random.permutation(len(mass_new))

mass_loss_rate = mass_new[shuffled_indices]
Imp_new = Imp_new.T
Imp1_500 = Imp_new[shuffled_indices]
original_indices_shuffled = original_indices[shuffled_indices]

# --------------------- 数据整理 -------------------
sequence_length = Imp1_500.shape[1]
X = np.array(Imp1_500)
y = np.array(mass_loss_rate)

X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std

split_time = int(0.7 * len(X))
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
train_indices = original_indices_shuffled[:split_time]
remaining_indices = original_indices_shuffled[split_time:]

X_val, X_test = X_remaining[:int(len(X_remaining) // 1.1)], X_remaining[int(len(X_remaining) // 1.1):]
y_val, y_test = y_remaining[:int(len(y_remaining) // 1.1)], y_remaining[int(len(y_remaining) // 1.1):]
val_indices = remaining_indices[:int(len(X_remaining) // 1.1)]
test_indices = remaining_indices[int(len(X_remaining) // 1.1):]

# --------------------- 注意力机制 -------------------
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

# --------------------- 改进后的残差块（嵌套注意力） -------------------
def residual_block(x, filters, kernel_size, pooling=True):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)

    # 在残差相加前插入注意力
    x = ChannelAttention(x)
    x = TemporalAttention(x)

    x = Add()([shortcut, x])
    x = LayerNormalization()(x)
    if pooling:
        x = MaxPooling1D(pool_size=2)(x)
    return x

# --------------------- 构建模型 -------------------
def build_advanced_model(input_length):
    inputs = Input(shape=(input_length, 1))
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)

    x = residual_block(x, 128, 3)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    return model

model = build_advanced_model(sequence_length)

optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.9, patience=64, min_lr=7.5e-4, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=96, restore_best_weights=True, verbose=1
)
callbacks = [lr_scheduler, early_stop]

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# --------------------- 测试集评估 --------------------
y_test_pred = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae = mean_absolute_error(y_test.squeeze(), y_test_pred)
test_absolute_errors = np.abs(y_test.squeeze() - y_test_pred.squeeze())
epsilon = 1e-10
test_relative_errors = np.abs((y_test.squeeze() - y_test_pred.squeeze()) / (y_test.squeeze() + epsilon)) * 100

print(f'''
=== 性能报告 ===
训练样本数：{len(y_train)}，测试样本数：{len(y_test)}
训练集 MAE: {train_mae:.4f}
测试集 MAE: {test_mae:.4f}
误差增加: {((test_mae - train_mae) / train_mae) * 100:.1f}%
绝对误差范围: [{np.min(test_absolute_errors):.4f}, {np.max(test_absolute_errors):.4f}]
平均绝对误差: {np.mean(test_absolute_errors):.4f}
相对误差范围: [{np.min(test_relative_errors):.2f}%, {np.max(test_relative_errors):.2f}%]
平均相对误差: {np.mean(test_relative_errors):.2f}%
''')

# --------------------- 绘图 --------------------
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
idx_test = test_indices
plot_idx = np.arange(len(y_test))

ax1.plot(plot_idx, y_test.squeeze()*100, 'bo-', label='True Value', markersize=4)
ax1.plot(plot_idx, y_test_pred*100, 'rx--', label='Predicted Value', markersize=4)
ax1.set_title('Test Set Prediction Comparison', fontweight='bold')
ax1.set_ylabel('Mass Loss Rate (%)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 25)
ax1.set_xlim(-0.5, len(y_test)-0.5)
ax1.set_xticks(plot_idx)
ax1.set_xticklabels(idx_test)

ax2.bar(plot_idx, test_absolute_errors*100, color='orange', alpha=0.7, label='Absolute Error')
ax2.set_title('Absolute Errors', fontweight='bold')
ax2.set_ylabel('Absolute Error (%)')
ax2.legend()
ax2.grid(True)
ax2.set_xticks(plot_idx)
ax2.set_xticklabels(idx_test)

plates = [f"P{i}" for i in range(1, 31) if i not in [4,6,18]]
cycles = [f"T{t}" for t in range(1, 5)]
signal_groups = [chr(ord('A') + i) for i in range(16)]

mapping = []
for sg in signal_groups:
    for cyc in cycles:
        for plate in plates:
            mapping.append(f"{plate}{cyc}{sg}")

assert len(mapping) == 1728, f"映射长度不对: {len(mapping)}"

converted_labels = [mapping[i] for i in idx_test]
for x, label in zip(plot_idx, converted_labels):
    ax2.text(x, -max(test_absolute_errors)*100*0.1,
             label, rotation=45, ha='right', va='top', rotation_mode='anchor', fontsize=9)

plt.tight_layout()
plt.show()
