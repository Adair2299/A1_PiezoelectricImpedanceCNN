# 0411可行
# 最相关:013B
# 对1501维进行了简化
"""
少01.17 6号钢板，现用5号代替
"""

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

# ------------识别文件1：质量损失------------
# 指定路径和文件
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\实验数据汇总\质量损失数据")
file = "质量统计.xlsx"

# 构建安全路径（自动处理操作系统差异）
file_path = folder / file

# 检查文件是否存在
if not file_path.exists():
    raise FileNotFoundError(f"文件不存在: {file_path}")

# 读取Excel数据
df = pd.read_excel(file_path, header=None, skiprows=1, usecols=[1, 2])  # 读取B,C列
mass = df.values.T.reshape(-1, 1)
mass_loss = 287 - mass  # 计算质量损失 (y)

# ---------识别文件2：阻抗谱 Piezoelectric Impedance Spectrum---------
# 提取term1
# 定义目标文件夹路径
folder_paths = [
    r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\20_50_Term1",
    r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\20_50_Term2"
]

Imp20_50_Orgl = []

# 遍历30个文件
for folder_path in folder_paths:
    for i in range(1, 31):  # 30 个文件
        filename = f"20-50 12.24_{i:02d}.xlsx" if "Term1" in folder_path else f"20-50 01.17_{i:02d}.xlsx"
        file_path = os.path.join(folder_path, filename)

        # 读取Excel的B列数据（B2-B2497对应索引1-1501）
        df = pd.read_excel(file_path, usecols="B", header=None, skiprows=1, nrows=1501)
        # 转换为向量并添加到矩阵
        Imp20_50_Orgl.append(df.iloc[:, 0].values)

# 转换为 numpy 数组（60行×1501列）
Imp20_50_Orgl = np.array(Imp20_50_Orgl)

"""
# 法1：直接删数据
Imp1_500 = Imp1_500_Orgl[:, range(2000)] # 取前2000个数
"""

"""
# 法2：求和削峰
divider = 2 # 相邻（）个一求和
Imp1_500 = Imp1_500_Orgl.reshape(60, int(1501 / divider), divider).sum(axis=2)

"""

# --------------------- 第1步：整理数据 --------------------
Imp20_50 = np.array(Imp20_50_Orgl)

sequence_length = Imp20_50.shape[1]  # 用1501个点预测mass_loss

X = np.array(Imp20_50)  # 阻抗数据
y = np.array(mass_loss)  # 质量损失

# **修改点：仅标准化 X，y 直接保持物理值**
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std  # 标准化 X
# y 保持原始尺度 (y 的单位是质量损失，不能归一化)

# **严格按时间顺序划分数据集**
split_time = int(0.7 * len(X))  # 70% 训练
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:] # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

# **转换为3D格式** [样本数, 时间步, 特征]
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# --------------------- 第2步：构建容易过拟合的模型 --------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(sequence_length, 1)),
    Conv1D(128, 3, activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1)  # 直接预测质量损失
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=8, # 一次用几个训练
    validation_data=(X_val, y_val),
    verbose=1 # 进度条
)

# **绘制训练过程**
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.pause(0.1)

# --------------------- 第4步：测试集评估（关键步骤） --------------------
# 预测时需要保持数据维度
y_test_pred = model.predict(X_test).squeeze()  # 预测并转换形状
y_train_pred = model.predict(X_train).squeeze()

# **计算误差**
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

print(f'''
=== 性能报告 ===
训练样本数：{len(y_train)}，测试样本数：{len(y_test)}
训练集MAE: {train_mae:.4f} （模型见过的数据）
测试集MAE: {test_mae:.4f} （全新数据）
误差增加: {((test_mae - train_mae) / train_mae) * 100:.1f}%
''')

# **绘制真实值 vs. 预测值**
plt.figure(figsize=(5, 5))
plt.plot(y_test, 'bo-', label='True Value')
plt.plot(y_test_pred, 'rx--', label='Predicted Value')
plt.title(f'Test Set Prediction Comparison ({len(y_test)} Samples)')
plt.legend()
plt.grid(True)

plt.ylim(0,25)
plt.show()

