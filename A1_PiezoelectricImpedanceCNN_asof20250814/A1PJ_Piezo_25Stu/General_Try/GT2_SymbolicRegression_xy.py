import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import pandas as pd

# 定义文件路径（注意Windows路径需要使用原始字符串或双反斜杠）
path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term3 1-500 03.06.xlsx"


# 或者
# path = "E:\\01我的\\大三下(202501-202508)\\大创-压电阻抗\\14天阻抗汇总\\Term3 1-500 03.06.xlsx"

# 精确读取A2:A2497作为X，Z2:Z2497作为Y
# # 方法1：使用openpyxl引擎直接读取指定范围
# def read_excel_ranges(path):
#     # 读取A列数据
#     df_A = pd.read_excel(path, sheet_name=0, header=None,
#                          usecols='A', skiprows=1, nrows=2496)  # A2:A2497
#     # 读取Z列数据
#     df_Z = pd.read_excel(path, sheet_name=0, header=None,
#                          usecols='Z', skiprows=1, nrows=2496)  # Z2:Z2497
#
#     X = df_A.values.reshape(-1)
#     y = df_Z.values.reshape(-1)
#     return X, y


# 方法2：更精确的读取方式（确保行列对应）
import pandas as pd
import numpy as np


def read_and_filter_excel(path):
    """
    读取Excel数据并删除指定区间

    参数:
        path: Excel文件路径

    返回:
        X_clean: 处理后的特征数据(保留好的数据点)
        y_clean: 处理后的目标数据(保留好的数据点)
        bad_indices: 被删除的索引位置(用于检查)
    """
    # 读取原始数据
    df = pd.read_excel(path, sheet_name=0, header=None,
                       usecols='A,Z',  # 同时读取A列和Z列
                       skiprows=1,  # 跳过标题行
                       nrows=2496)  # 读取2496行(A2:A2497和Z2:Z2497)

    X = df.iloc[:, 0].values  # 第一列(A列)
    y = df.iloc[:, 1].values  # 第二列(Z列)

    # 定义要删除的区间(左闭右开)
    bad_ranges = [
        (250, 400),
        (850, 1000),
        (1400, 1550),
        (1950, 2130)
    ]

    # 生成所有需要删除的索引
    bad_indices = []
    for start, end in bad_ranges:
        bad_indices.extend(range(start, end))

    # 创建掩码(True表示保留)
    mask = np.ones(len(X), dtype=bool)
    mask[bad_indices] = False

    # 应用掩码
    X_clean = X[mask]
    y_clean = y[mask]

    # 转换为二维数组(符号回归需要)
    X_clean = X_clean.reshape(-1, 1)

    return X_clean, y_clean, bad_indices


# 使用示例
path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term3 1-500 03.06.xlsx"
X, y, removed_indices = read_and_filter_excel(path)

# # 检查结果
# print(f"原始数据量: 2496, 处理后数据量: {len(X_clean)}")
# print(f"删除数据量: {len(removed_indices)}")
# print(f"X_clean前5个值:\n{X_clean[:5]}")
# print(f"y_clean前5个值:\n{y_clean[:5]}")
#

# 检查数据
print(f"X shape: {X.shape}, 前5个值: {X[:5]}")
print(f"y shape: {y.shape}, 前5个值: {y[:5]}")

# 确保没有NaN值
assert not np.isnan(X).any(), "X包含NaN值!"
assert not np.isnan(y).any(), "y包含NaN值!"

# 转换为适合符号回归的格式(二维数组)
X = X.reshape(-1, 1)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 创建符号回归模型
est_gp = SymbolicRegressor(
    population_size=5000,  # 种群大小
    generations=20,       # 进化代数
    tournament_size=20,   # 锦标赛选择的大小
    stopping_criteria=0.01,  # 停止标准(如果适应度小于此值则停止)
    const_range=(-1, 1),  # 常数范围
    init_depth=(2, 6),    # 初始树的深度范围
    init_method='half and half',  # 初始化方法
    function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt'),  # 使用的函数
    # 可选择：'add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'
    metric='rmse',        # 适应度指标
    parsimony_coefficient=0.001,  # 简约系数(防止过拟合)
    p_crossover=0.7,      # 交叉概率
    p_subtree_mutation=0.1,  # 子树变异概率
    p_hoist_mutation=0.05,   # 提升变异概率
    p_point_mutation=0.1,    # 点变异概率
    max_samples=0.9,      # 用于拟合的样本比例
    verbose=1,            # 显示进度
    random_state=42       # 随机种子
)

# 3. 训练模型
est_gp.fit(X_train, y_train)

# 4. 评估模型
print(f"\n最佳程序:\n{est_gp._program}")
print(f"训练集RMSE: {np.sqrt(mean_squared_error(y_train, est_gp.predict(X_train))):.4f}")
print(f"测试集RMSE: {np.sqrt(mean_squared_error(y_test, est_gp.predict(X_test))):.4f}")



from sympy import symbols, simplify, latex
import numpy as np


def beautify_expression(gp_program, var_name='x'):
    """
    将gplearn的表达式转换为美观的数学形式

    参数:
        gp_program: gplearn的_program对象
        var_name: 变量名(默认为'x')

    返回:
        dict: 包含不同格式的表达式
    """
    x = symbols(var_name)

    # 转换函数映射
    func_map = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b,
        'sin': lambda a: sin(a),
        'cos': lambda a: cos(a),
        'log': lambda a: log(a),
        'sqrt': lambda a: sqrt(a),
        'neg': lambda a: -a,
        'inv': lambda a: 1 / a
    }

    # 递归解析表达式树
    def parse_node(node):
        if isinstance(node, float) or isinstance(node, int):
            return node
        elif isinstance(node, str):  # 处理常数项
            try:
                return float(node)
            except:
                return x
        else:
            func = func_map[node.name]
            args = [parse_node(arg) for arg in node.programs]
            return func(*args)

    # 解析并简化表达式
    raw_expr = parse_node(gp_program)
    simplified = simplify(raw_expr)

    return {
        'raw': str(raw_expr),
        'simplified': str(simplified),
        'latex': latex(simplified),
        'pretty': simplified
    }


# 使用示例
gp_expression = est_gp._program  # 你的符号回归结果
result = beautify_expression(gp_expression)

print("=== 表达式美化结果 ===")
print(f"原始表达式: {result['raw']}")
print(f"简化表达式: {result['simplified']}")
print(f"LaTeX格式: {result['latex']}")




# 5. 可视化结果
plt.figure(figsize=(12, 6))

# 绘制原始函数
x_plot = np.linspace(0, 500000, 200).reshape(-1, 1)
y_true = y
plt.scatter(X, y_true, label='Orginal Data')

# 绘制训练数据
plt.scatter(X_train, y_train, c='g', label='Training Data', alpha=0.6)

# 绘制预测结果
y_pred = est_gp.predict(x_plot)
plt.plot(x_plot, y_pred, 'r-', label='Prediction of Symbolic Regression', linewidth=2)

plt.title('Result of Symbolic Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()