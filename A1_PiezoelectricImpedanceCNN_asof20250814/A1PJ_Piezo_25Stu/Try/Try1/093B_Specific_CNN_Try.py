#  原代码见2A_CNN_Evaluation.py


# 一维CNN完整示例：用历史股价预测未来价格
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import pandas as pd
import os
from pathlib import Path

# ------------识别文件1：质量损失------------
# 指定路径和文件（关键：使用原始字符串 r"" 和 Path 库）
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\实验数据汇总\质量损失数据")
file = "质量统计.xlsx"

# 构建安全路径（自动处理操作系统差异）
file_path = folder / file

# 检查文件是否存在
if not file_path.exists():
    raise FileNotFoundError(f"文件不存在: {file_path}")

# 读取Excel数据
df = pd.read_excel(
    file_path,
    header=None,
    skiprows=1,
    usecols=[1]  # 读取B列
)

mass_loss = df.values.tolist()

# ---------识别文件2：阻抗谱 Piezoelectric Impedance Spectrum---------

# 定义目标文件夹路径
folder_path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term1"

# 创建空矩阵容器
Imp1_500 = []

# 遍历30个文件
for i in range(1, 31):
    # 生成带两位序号的文件名（01-30）
    filename = f"1-500 12.24_{i:02d}.xlsx"
    file_path = os.path.join(folder_path, filename)

    # 读取Excel的B列数据（B2-B2497对应索引1-2496）
    df = pd.read_excel(file_path, usecols="B", header=None, skiprows=1, nrows=2496)

    # 转换为向量并添加到矩阵
    Imp1_500.append(df.iloc[:, 0].values)

# 转换为numpy矩阵（30行×2496列）
Imp1_500 = np.array(Imp1_500)

# --------------------- 第1步：生成模拟数据 --------------------
# def generate_time_series(size=1000, seq_len=20):
#     """生成带噪声的正弦波时序数据"""
#     t = np.linspace(0, 10, size)
#     data = np.sin(t) + np.random.normal(0, 0.2, size)  # 正弦波+噪声
#     return data
#
# # 生成数据并创建输入输出对
# data = generate_time_series()
X, y = [], []
sequence_length = 2496  # 用过去20个点预测下一个点
# for i in range(len(data) - sequence_length):
#     X.append(data[i:i+sequence_length])
#     y.append(data[i+sequence_length])

X = np.array(Imp1_500)
y = np.array(mass_loss)

# 标准化
mean, std = X.mean(), X.std()
X = (X - mean) / std
y = (y - mean) / std

# 严格按时间顺序划分数据集（防止数据泄漏）
split_time = int(0.7 * len(X))  # 70%训练, 15%验证, 15%测试
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining)//2], X_remaining[len(X_remaining)//2:]
y_val, y_test = y_remaining[:len(y_remaining)//2], y_remaining[len(y_remaining)//2:]

# 转换为三维输入 [样本, 时间步, 特征]
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# --------------------- 第2步：构建容易过拟合的模型 --------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(sequence_length, 1)),
    Conv1D(128, 3, activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

# 绘制训练曲线
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# --------------------- 第4步：测试集评估（关键步骤） 092A--------------------
# 注意：运行前需要确保X已转换为3D格式 (样本数, 时间步长, 特征数)
# 添加数据维度检查
print("X_test形状验证:", X_test.shape)  # 应为 (n_samples, sequence_length, 1)

# 预测时需要保持数据维度
y_test_pred = model.predict(X_test)  # 输出形状为 (n_samples, 1)
y_test_pred = y_test_pred.squeeze()  # 从 (n,1) 转换为 (n,)

# 处理训练集预测结果
y_train_pred = model.predict(X_train).squeeze()

# 计算误差指标
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

print(f'''
=== 性能报告 ===
训练样本数：{len(y_train)}，测试样本数：{len(y_test)}
训练集MAE: {train_mae:.4f} （模型见过的数据）
测试集MAE: {test_mae:.4f} （全新数据）
结论：测试集误差比训练集高 {((test_mae - train_mae)/train_mae)*100:.1f}% 
''')

# 可视化调整（适应小样本数据）
plt.figure(figsize=(10,4))
plt.plot(y_test, 'bo-', label='True Value')
plt.plot(y_test_pred, 'rx--', label='Predicted Value')
plt.title(f'Test set prediction comparison ({len(y_test)} Samples)')
plt.legend()
plt.grid(True)
plt.show()