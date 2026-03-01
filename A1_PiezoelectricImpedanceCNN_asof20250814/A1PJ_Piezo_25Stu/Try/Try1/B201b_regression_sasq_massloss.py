# 指定公式回归预测 （我现在假定一个y=a*x1^b+m*x3^n，我欲求出最优的a b m n）
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
np.random.seed(42)  # 可选：设置随机种子（保证结果可复现）
shuffled_indices = np.random.permutation(len(mass_loss_rate))

# 使用相同的索引打乱两个矩阵
mass_loss_rate = mass_loss_rate[shuffled_indices]
sasq = sasq[shuffled_indices]


##########################################################
# -------------------- 指定公式回归开始 ---------------------


"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =================
plt.figure()
plt.scatter(sasq[:,0],mass_loss_rate[:,0])
plt.show()
# =================

# 定义目标函数（仅使用x₁和x₃）
def model_func(X, a, b, m, n):
    x1, x3 = X[:, 0], X[:, 1]  # X应为两列：[x₁, x₃]
    return a * x1**b + m * x3**n

# 加载数据（假设sasq.shape=(145,4), mass_loss_rate.shape=(145,)）
X_data = sasq[:, [0, 2]]  # 只取第0列(x₁)和第2列(x₃)
y_data = mass_loss_rate.flatten()

# 初始参数猜测（重要！根据物理意义调整）
initial_guess = [0.07, 2.0, 0.06, 1.5]  # [a, b, m, n]

# 拟合模型
params_opt, _ = curve_fit(
    lambda X, a, b, m, n: model_func(X, a, b, m, n),
    X_data, y_data,
    p0=initial_guess,
    maxfev=10000
)

# 输出最优参数
a, b, m, n = params_opt
print(f"最优参数: a={a:.4f}, b={b:.4f}, m={m:.4f}, n={n:.4f}")

# 评估拟合效果
y_pred = model_func(X_data, a, b, m, n)
relative_errors = np.abs((y_data - y_pred) / y_data) * 100
print(f"平均相对误差: {np.mean(relative_errors):.2f}%")

# 可视化
plt.figure(figsize=(10, 5))
plt.scatter(y_data, y_pred, alpha=0.6, label="Predicted values")
plt.plot([y_data.min(), y_data.max()], [y_data.min(), y_data.max()],
         'r--', label="Perfect fit")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title(f"Fitting results: $y = {a:.2f}x_1^{{{b:.2f}}} + {m:.2f}x_3^{{{n:.2f}}}$")
plt.legend()
plt.grid()
plt.show()
"""