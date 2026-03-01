# 符号回归Symbolic Regression
# 4月28，不可行，mass_loss_rate和sasq矩阵（向量）正确
# 预测不可视化
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
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split

# 假设 sasq 是 (145, 4) 的数组，mass_loss_rate 是 (145,) 的数组
X = sasq  # x1, x2, x3, x4
y = mass_loss_rate  # x5

# 分割训练集和测试集（可选）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义符号回归模型
est_gp = SymbolicRegressor(
    population_size=5000,  # 更大的种群有助于找到更优解
    generations=20,        # 迭代次数
    stopping_criteria=0.01, # 如果误差 < 0.01 就停止
    p_crossover=0.7,       # 交叉概率
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,       # 使用 90% 的数据训练
    verbose=1,
    parsimony_coefficient=0.01,  # 防止公式过于复杂
    random_state=42,
    function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt')  # 允许的函数
)

# 训练模型
est_gp.fit(X_train, y_train)

# 打印最佳公式
print("最佳拟合公式:", est_gp._program)

# 预测新数据
# new_data = np.array([[x1, x2, x3, x4]])  # 替换为你的新数据
# predicted_x5 = est_gp.predict(new_data)
# print("预测 x5:", predicted_x5[0])