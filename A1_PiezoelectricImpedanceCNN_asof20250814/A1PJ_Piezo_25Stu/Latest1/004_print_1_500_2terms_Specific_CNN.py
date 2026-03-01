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
from matplotlib import rcParams


# ------------识别文件1：质量损失------------
# 指定路径和文件
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作")
file = "3_sasq和质量损失率.xlsx"

# 构建安全路径（自动处理操作系统差异）
file_path = folder / file

# 检查文件是否存在
if not file_path.exists():
    raise FileNotFoundError(f"文件不存在: {file_path}")

# 读取Excel数据
# 读取 B16:AE16 和 B24:AE24 两行数据
df1 = pd.read_excel(file_path, header=None, skiprows=15, nrows=1, usecols=range(1, 31))

# 读取第二组数据(B24:AE24)
df2 = pd.read_excel(file_path, header=None, skiprows=23, nrows=1, usecols=range(1, 31))
df3 = pd.read_excel(file_path, header=None, skiprows=31, nrows=1, usecols=range(1, 31))
df4 = pd.read_excel(file_path, header=None, skiprows=39, nrows=1, usecols=range(1, 31))

# 合并两个DataFrame
result = pd.concat([df1, df2, df3, df4], ignore_index=True)
result = np.array(result)
result = np.delete(result, 5, axis=1) # 6号钢板数据不要
mass_loss_rate = result.reshape(-1, 1)


# ---------识别文件2：阻抗谱 Piezoelectric Impedance Spectrum---------

# 定义文件路径
import pandas as pd
from pathlib import Path

# 文件路径
file_path = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term all 1-500.xlsx")

# 精确读取 B2:DM2497
try:
    df = pd.read_excel(file_path, usecols="B:DM", header=None, skiprows=1, nrows=2496)
    print(f"成功读取，形状为: {df.shape}")  # 应该是 (2496, 117)

except Exception as e:
    print(f"读取失败: {e}")

Imp1_500_Orgl = df.values
print(f"最终数组形状: {Imp1_500_Orgl.shape}")  # 应该是(116, 2496)

Imp1_500 = np.array(Imp1_500_Orgl)

# --------打乱--------
# 生成相同的随机排列索引
np.random.seed(2)  # 可选：设置随机种子（保证结果可复现）
shuffled_indices = np.random.permutation(len(mass_loss_rate))

# 使用相同的索引打乱两个矩阵
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]



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
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:] # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

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

model = Sequential([
    # 第一层卷积层
    Conv1D(128, 8, activation='relu', input_shape=(sequence_length, 1)),  # 使用更大的卷积核来提取更复杂的特征
    MaxPooling1D(2),  # 池化层，减少序列长度
    # # 第二层卷积层
    # Conv1D(64, 4, activation='relu'),
    # MaxPooling1D(2),
    # # 第三层卷积层
    # Conv1D(32, 2, activation='relu'),
    Flatten(),
    # 全连接层
    Dense(64, activation='relu'),
    Dropout(0.3),  # Dropout防止过拟合
    Dense(1)  # 输出层，预测质量损失
])


model.compile(optimizer='adam', loss='mse')
model.summary()

# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=500, # 迭代次数
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
plt.ylim(0,0.006)
plt.pause(0.1)




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


from datetime import datetime
from matplotlib.gridspec import GridSpec


# --------------------- 生成时间戳 --------------------
current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # 格式如：20250808223405
model_folder_name = f"最初CNN模型{current_time}"

# --------------------- 打印性能报告 --------------------
performance_report = f'''
=== Performance Report ===

Training samples: {len(y_train)}, Test samples: {len(y_test)}
Training MAE: {train_mae:.4f}
Test MAE: {test_mae:.4f}
Error increase: {((test_mae - train_mae) / train_mae) * 100:.1f}%

--- Detailed Test Set Analysis ---

Absolute error range: [{np.min(test_absolute_errors):.4f}, {np.max(test_absolute_errors):.4f}]
Mean absolute error: {np.mean(test_absolute_errors):.4f}

Relative error range: [{np.min(test_relative_errors):.2f}%, {np.max(test_relative_errors):.2f}%]
Mean relative error: {np.mean(test_relative_errors):.2f}%
'''

print(performance_report)

# --------------------- 绘图设置 --------------------
# 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10


# --------------------- 保存高分辨率图像 --------------------
output_folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\论文材料\消融研究") / model_folder_name

# 确保输出目录存在
output_folder.mkdir(parents=True, exist_ok=True)

# 1. 保存训练过程图
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=0.5)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=0.5)
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.ylim(0, 0.01)
plt.legend()
plt.savefig(output_folder / "training_process.png", dpi=600, bbox_inches='tight')
plt.close()  # 关闭图形，避免重叠

# 2. 保存预测对比和误差图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

# 子图1：真实值 vs 预测值
sample_numbers = np.arange(1, len(y_test) + 1)
ax1.plot(sample_numbers, y_test * 100, 'bo-', label='True Value', markersize=4, linewidth=0.5)
ax1.plot(sample_numbers, y_test_pred * 100, 'rx--', label='Predicted Value', markersize=4, linewidth=0.5)
ax1.set_title('Test Set Prediction Comparison', fontweight='bold')
ax1.set_ylabel('Mass Loss Rate (%)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 25)
ax1.set_xlim(0.5, len(y_test) + 0.5)

# 子图2：绝对误差
ax2.bar(sample_numbers, test_absolute_errors * 100, color='orange', alpha=0.7,
       label='Absolute Error', width=0.6)
ax2.set_title('Absolute Errors', fontweight='bold')
ax2.set_xlabel('Sample Number')
ax2.set_ylabel('Absolute Error (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(output_folder / "prediction_comparison.png", dpi=600, bbox_inches='tight')
plt.close()  # 关闭图形

# 3. 保存性能报告为图片
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])

# 隐藏坐标轴
ax.axis('off')

# 添加文本
ax.text(0.1, 0.9, 'Model Performance Report', fontsize=14, fontweight='bold')
ax.text(0.1, 0.8, performance_report, fontsize=10, va='top', ha='left')

plt.tight_layout()
plt.savefig(output_folder / "performance_report.png", dpi=600, bbox_inches='tight')
plt.close()

print(f"图像已保存至: {output_folder}")