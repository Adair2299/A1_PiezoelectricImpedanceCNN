# 04月13日可行
# 20-50频谱,30天
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

# ------------识别文件1：质量损失------------
# 指定路径和文件
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总")
file = "14质量统计.xlsx"

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

# 定义文件路径
file_paths = [
    r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term1 20-50 12.24.xlsx",
    r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term2 20-50 01.17.xlsx"
]

# 初始化结果列表
Imp20_50_Orgl = []

for file_path in file_paths:
    try:
        # 读取Excel文件，B到AD列（共29列），跳过第一行标题，读取1501行数据
        # 注意：usecols="B:AD"表示从B列到AD列（包含AD列）
        df = pd.read_excel(file_path, usecols="B:AD", header=None, skiprows=1, nrows=1501)

        # 将DataFrame转换为numpy数组并添加到结果列表
        Imp20_50_Orgl.append(df.values)

        print(f"成功处理文件: {file_path}")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        # 如果文件读取失败，可以添加一个NaN数组保持形状一致
        Imp20_50_Orgl.append(np.full((1501, 29), np.nan))

# 将两个数组合并为一个58×1501的数组
# 首先将两个29×1501的数组转置为1501×29，然后垂直堆叠
Imp20_50_Orgl = np.vstack([arr.T for arr in Imp20_50_Orgl])

print(f"最终数组形状: {Imp20_50_Orgl.shape}")  # 应该是(58, 1501)
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

