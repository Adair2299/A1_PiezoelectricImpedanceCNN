# 神经网络预测
# 4月29日，可行，mass_loss_rate和sasq矩阵（向量）正确
# 预测可视化，效果不错
# 找sa,sq,与massloss的关系

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')  # 解决plt警告
import pandas as pd
from pathlib import Path

# ------------识别文件1：质量损失------------
# 指定路径和文件
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作")
file = "3_sasq和质量损失率.xlsx"

# 构建安全路径（自动处理操作系统差异）
file_path = folder / file


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
mass_loss_rate = result.reshape(-1, 1) # 116行*1列

# 垂直拼接（前面加 29 个 0），补充第0天未腐蚀的关系
zeros_29 = np.zeros((29, 1))
mass_loss_rate = np.concatenate((zeros_29, mass_loss_rate), axis=0)

# ------------识别文件2：sa和sq------------
# 读取Excel文件
file_path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作\1_sasq数据整合.xlsx"
df = pd.read_excel(file_path, header=None)  # 假设没有标题行

# 初始化空列表来存储提取的数据
sasq_orgl = []

# 定义起始行号（注意：Python是0-based索引，Excel行号是1-based）
start_rows = [1, 5, 9]  # 对应Excel中的B2, B6, B10行
end_row = 37  # 对应Excel中的B38行

# 循环提取数据，每次增加4行（隔两行取两行）
current_row = 1  # 从B2开始（Python索引1对应Excel第2行）
while current_row <= end_row:
    # 提取两行数据（B列到AE列，即Python索引1到30）
    data = df.iloc[current_row:current_row+2, 1:31].values
    sasq_orgl.extend(data.tolist())
    current_row += 4  # 跳过两行

sasq_orgl = np.array(sasq_orgl)
sasq_orgl = np.delete(sasq_orgl, 5, axis=1) # 6号钢板数据不要,20行*29列

# 初始化结果矩阵（4行×145列）
sasq = np.zeros((4, 145))

# 遍历4个目标行
for i in range(4):
    # 每隔4行取1行（i, i+4, i+8, i+12, i+16）
    rows = [sasq_orgl[i + j*4, :] for j in range(5)]  # 5组数据
    # 横向拼接
    sasq[i, :] = np.hstack(rows) # 4行*145列
sasq = sasq.T # 145行*4列

# --------打乱--------
# 生成相同的随机排列索引
np.random.seed(42)  # 可选：设置随机种子（保证结果可复现）
shuffled_indices = np.random.permutation(len(mass_loss_rate))

# 使用相同的索引打乱两个矩阵
mass_loss_rate = mass_loss_rate[shuffled_indices]
sasq = sasq[shuffled_indices]


##########################################################
# -------------------- 神经网络开始 ----------------------
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# --------------------- 第1步：数据准备 --------------------
# 假设已经加载数据：
# sasq = np.array(...)  # 形状 (145, 4)
# mass_loss_rate = np.array(...)  # 形状 (145, 1)

# 仅标准化X，y保持物理值
X_scaler = StandardScaler()
X = X_scaler.fit_transform(sasq)  # 标准化X (x1-x4)
y = mass_loss_rate  # y保持原始值

# 严格按时间顺序划分数据集
split_time = int(0.7 * len(X))  # 70% 训练
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining)//2], X_remaining[len(X_remaining)//2:]  # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:len(y_remaining)//2], y_remaining[len(y_remaining)//2:]

# --------------------- 第2步：构建模型 --------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),  # 输入层：4个特征
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dense(1)  # 直接预测质量损失
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=12,
    validation_data=(X_val, y_val),
    verbose=1
)


# --------------------- 第4步：测试集评估 --------------------
y_test_pred = model.predict(X_test).flatten()  # 预测并转换形状
y_train_pred = model.predict(X_train).flatten()

# 只计算测试集的绝对误差（不再计算训练集误差）
test_mae = mean_absolute_error(y_test, y_test_pred)  # 测试集MAE
absolute_errors = np.abs(y_test - y_test_pred)       # 每个样本的绝对误差
mean_abs_error = np.mean(absolute_errors)            # 平均绝对误差
max_abs_error = np.max(absolute_errors)              # 最大绝对误差


print(f'''
=== 测试集性能报告 ===
测试样本数量: {len(y_test)}
MAE (平均绝对误差): {test_mae:.4f}
平均绝对误差 (逐点计算): {mean_abs_error:.4f}
最大绝对误差: {max_abs_error:.4f}
''')


# 绘制训练过程
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# 绘制真实值 vs. 预测值
plt.figure(figsize=(5, 5))
plt.plot(y_test, 'bo-', label='True Value')
plt.plot(y_test_pred, 'rx--', label='Predicted Value')
plt.title(f'Test Set Prediction Comparison ({len(y_test)} Samples)')
plt.legend()
plt.grid(True)
plt.show()

# # 预测新数据
# def predict(x1, x2, x3, x4):
#     new_data = np.array([[x1, x2, x3, x4]])
#     new_data_scaled = X_scaler.transform(new_data)  # 标准化输入
#     predicted = model.predict(new_data_scaled)[0][0]  # 直接输出物理值
#     return predicted
#
# # 示例预测
# x1, x2, x3, x4 = 1.0, 2.0, 3.0, 4.0  # 替换为你的输入
# predicted_x5 = predict(x1, x2, x3, x4)
# print(f"预测值 x5: {predicted_x5}")