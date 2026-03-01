# 符号回归Symbolic Regression
# 4月28，不可行， mass_loss_rate和sasq矩阵（向量）正确
# 预测可视化尝试
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
np.random.seed(2)  # 可选：设置随机种子（保证结果可复现）
shuffled_indices = np.random.permutation(len(mass_loss_rate))

# 使用相同的索引打乱两个矩阵
mass_loss_rate = mass_loss_rate[shuffled_indices]
sasq = sasq[shuffled_indices]


##########################################################
# --------------------
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==================== 数据准备 ====================
X = sasq
y = mass_loss_rate.flatten()  # 确保 y 是一维的 (145,)

# 标准化 X（可选）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==================== 符号回归建模 ====================
est_gp = SymbolicRegressor(
    population_size=10000,  # 增加种群大小
    generations=50,  # 增加迭代次数
    stopping_criteria=0.001,  # 放宽停止准则
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=42,
    function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt')  # 移除exp
)

est_gp.fit(X_train, y_train)

# ==================== 评估与可视化 ====================
# 预测训练集和测试集
y_train_pred = est_gp.predict(X_train)
y_test_pred = est_gp.predict(X_test)

# 计算R²分数
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"训练集R²: {train_r2:.4f}")
print(f"测试集R²: {test_r2:.4f}")
print("最佳拟合公式:", est_gp._program)

# ==================== 可视化1: 预测 vs. 真实值 ====================
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Training set')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', label='Perfect prediction')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'Training set: True vs Predicted (R²={train_r2:.3f})')
plt.legend()

# 测试集
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, color='orange', label='Test set')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect prediction')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'Test set: True vs Predicted (R²={test_r2:.3f})')
plt.legend()

plt.tight_layout()
plt.show()

# ==================== 可视化2: 残差分析 ====================
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(12, 5))

# 训练集残差
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, residuals_train, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Training set residuals')

# 测试集残差
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, residuals_test, alpha=0.5, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Test set residuals')

plt.tight_layout()
plt.show()

# ==================== 可视化3: 误差直方图 ====================
plt.figure(figsize=(8, 5))
plt.hist(residuals_train, bins=30, alpha=0.5, label='Training set')
plt.hist(residuals_test, bins=30, alpha=0.5, color='orange', label='Test set')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Prediction error')
plt.ylabel('Frequency')
plt.title('Prediction error distribution')
plt.legend()
plt.show()
