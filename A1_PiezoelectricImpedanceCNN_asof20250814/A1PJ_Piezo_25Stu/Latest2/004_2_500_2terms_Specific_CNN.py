# 04月27日可行,06月07日可行，神经网络预测massloss
# 1-500频谱，60天的， 四个周期全部
# 没用6号钢板，只用58组



import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # 解决plt警告
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
from pathlib import Path
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D


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
# result = np.delete(result, 5, axis=1) # 6号钢板数据不要
# mass_loss_rate = result.reshape(-1, 1)
#
#
# # ---------识别文件2：阻抗谱 Piezoelectric Impedance Spectrum---------
#
# # 定义文件路径
#
#
# # 文件路径
# file_path = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term all 1-500.xlsx")
# # file_path = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term all 1-500 GaussianSmoothed Sigma10.xlsx")
#
# # 精确读取 B2:DM2497
# try:
#     df = pd.read_excel(file_path, usecols="B:DM", header=None, skiprows=1, nrows=2496)
#     print(f"成功读取，形状为: {df.shape}")  # 应该是 (2496, 117)
#
# except Exception as e:
#     print(f"读取失败: {e}")
#
# Imp1_500_Orgl = df.values
# print(f"最终数组形状: {Imp1_500_Orgl.shape}")  # 应该是(2496, 116)
#
# Imp1_500 = np.array(Imp1_500_Orgl)
#

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

# plt.plot(Imp1_500[:, 84], label='Origin', alpha=0.5)
# plt.plot(Imp1_500_smooth[:, 84], label='Adaptive Smoothed', linewidth=2)
# plt.legend()
# plt.title('第1列阻抗谱：自适应平滑效果')
# plt.show()

# Imp1_500 = gaussian_filter(Imp1_500, sigma=[15,0]) # 简单平滑

# --------打乱--------
# 生成相同的随机排列索引
np.random.seed(2)  # 可选：设置随机种子（保证结果可复现）
shuffled_indices = np.random.permutation(len(mass_new))

# 使用相同的索引打乱两个矩阵
mass_loss_rate = mass_new[shuffled_indices]
Imp_new = Imp_new.T
Imp1_500 = Imp_new[shuffled_indices]



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
X_val, X_test = X_remaining[:int(len(X_remaining) // 1.1)], X_remaining[int(len(X_remaining) // 1.1):] # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:int(len(y_remaining) // 1.1)], y_remaining[int(len(y_remaining) // 1.1):]

# **转换为3D格式** [样本数, 时间步, 特征]
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# --------------------- 第2步：构建容易过拟合的模型 --------------------
# model = Sequential([
#     Conv1D(32, 4, activation='relu', input_shape=(sequence_length, 1)),
#     Conv1D(16, 2, activation='relu'),
#     Flatten(),
#     Dense(8, activation='relu'),
#     Dense(1)  # 直接预测质量损失
# ])

# model = Sequential([
#     # 第一层卷积层
#     Conv1D(128, 8, activation='relu', input_shape=(sequence_length, 1)),  # 使用更大的卷积核来提取更复杂的特征
#     MaxPooling1D(2),  # 池化层，减少序列长度
#     # 第二层卷积层
#     Conv1D(64, 4, activation='relu'),
#     MaxPooling1D(2),
#     # 第三层卷积层
#     Conv1D(32, 2, activation='relu'),
#     Flatten(),
#     # 全连接层
#     Dense(64, activation='relu'),
#     Dropout(0.3),  # Dropout防止过拟合
#     Dense(1)  # 输出层，预测质量损失
# ])


# LSTM模型示例
model = Sequential([
    Conv1D(64, 5, activation='relu', padding='same', input_shape=(156, 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),

    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])


# 1. 定义优化器和回调
optimizer = Adam(learning_rate=0.001)  # 初始学习率
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)
callbacks = [lr_scheduler, early_stop]  # 组合回调
model.compile(optimizer='adam', loss='mae')
model.summary()

# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=400, # 迭代次数
    batch_size=128, # 一次用几个训练
    validation_data=(X_val, y_val),
    # callbacks=callbacks,
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
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10




# **绘制训练过程**
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
# plt.ylim(0,0.006)
plt.pause(0.1)




# 创建包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

# --------------------- 子图1：真实值 vs 预测值 --------------------
sample_numbers = np.arange(1, len(y_test)+1)  # 2. 横坐标从1开始计数

ax1.plot(sample_numbers, y_test*100, 'bo-', label='True Value', markersize=4)
ax1.plot(sample_numbers, y_test_pred*100, 'rx--', label='Predicted Value', markersize=4)
ax1.set_title('Test Set Prediction Comparison', fontweight='bold')
ax1.set_ylabel('Mass Loss Rate (%)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 25)
ax1.set_xlim(0.5, len(y_test)+0.5)

# --------------------- 子图2：绝对误差 --------------------
ax2.bar(sample_numbers, test_absolute_errors*100, color='orange', alpha=0.7, label='Absolute Error')
ax2.set_title('Absolute Errors', fontweight='bold')
ax2.set_xlabel('Sample Number')
ax2.set_ylabel('Absolute Error (%)')
ax2.legend()
ax2.grid(True)

# 调整子图间距
plt.tight_layout()
plt.show()
