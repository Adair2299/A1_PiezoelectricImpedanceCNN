import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
np.random.seed(42)

# 1. 生成模拟数据
# 四个自变量X1, X2, X3, X4
n_samples = 1000
X1 = np.random.rand(n_samples) * 10 - 5  # -5到5之间均匀分布
X2 = np.random.rand(n_samples) * 8 - 4    # -4到4之间均匀分布
X3 = np.random.rand(n_samples) * 6 - 3    # -3到3之间均匀分布
X4 = np.random.rand(n_samples) * 4 - 2    # -2到2之间均匀分布

# 真实函数（这是我们希望符号回归能发现的）
# 这里我们使用一个稍微复杂的函数作为示例
y_true = 2.5 * X1 + np.sin(X2) * X3**2 - 0.5 * X4**3 + np.random.normal(0, 0.5, n_samples)

# 组合自变量
X = np.column_stack((X1, X2, X3, X4))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# 2. 创建符号回归模型
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20,
                           tournament_size=20,
                           stopping_criteria=0.01,
                           const_range=(-1, 1),
                           init_depth=(2, 6),
                           init_method='half and half',
                           function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'),
                           metric='mse',
                           parsimony_coefficient=0.001,
                           random_state=42,
                           n_jobs=-1,
                           verbose=1)

# 3. 训练模型
est_gp.fit(X_train, y_train)

# 4. 评估模型
y_pred = est_gp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n测试集MSE: {mse:.4f}")

# 5. 输出最佳表达式
print("\n最佳表达式:")
print(est_gp._program)

# 6. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted Values')  # Added label for scatter
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         'r--', label='Perfect Prediction Line')  # Added label for line
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('The Prediction Result of Symbolic Regression')
plt.legend()  # Corrected legend display (changed from legend.show to plt.legend())
plt.show()