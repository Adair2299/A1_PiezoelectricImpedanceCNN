# 神经网络预测我的基线假设
# 7月4日


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


# ------------识别文件2：假定方程基本参数------------
# 文件路径
file_path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作\10_1_基线假设方程全部参数汇总1.xlsx"

# 读取四个sheet的数据
term1 = pd.read_excel(file_path, sheet_name="Term1", header=None, skiprows=1, nrows=14, usecols="B:AD")
term2 = pd.read_excel(file_path, sheet_name="Term2", header=None, skiprows=1, nrows=14, usecols="B:AD")
term3 = pd.read_excel(file_path, sheet_name="Term3", header=None, skiprows=1, nrows=14, usecols="B:AD")
term4 = pd.read_excel(file_path, sheet_name="Term4", header=None, skiprows=1, nrows=14, usecols="B:AD")

# 将四个DataFrame水平拼接
combined_matrix = pd.concat([term1, term2, term3, term4], axis=1)

# 如果需要numpy数组
paras = combined_matrix.values

"""
变量准备结束，神经网络开始
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 设置随机种子保证可重复性
np.random.seed(42)
tf.random.set_seed(42)

# ------------数据准备------------
# 假设已经得到了 paras (14×116) 和 mass_loss_rate (116×1)
# 我们需要转置 paras 使其成为 116×14 的矩阵
X = paras.T  # 现在 X 是 116×14
y = mass_loss_rate  # 116×1

# 检查数据形状
print("X shape:", X.shape)
print("y shape:", y.shape)

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

# ------------数据打乱------------
# 显式打乱训练数据
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print("\n数据打乱检查（前5个样本的y值）:")
print(y_train[:5].flatten())

# ------------构建神经网络模型------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # 输出层，线性激活
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# 早停法回调
early_stop = EarlyStopping(monitor='val_loss',
                          patience=50,
                          verbose=1,
                          restore_best_weights=True)

# ------------训练模型------------
print("\n开始训练...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1,
    shuffle=True  # 每个epoch内部也会打乱batch顺序
)

# ------------模型评估------------
# 预测测试集
y_pred_scaled = model.predict(X_test)

# 反标准化
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

# 计算指标
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, y_pred)
print(f"\n评估结果:")
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")

# ------------可视化------------
plt.figure(figsize=(15, 10))

# 1. 训练历史
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()

# 2. 预测 vs 真实值
plt.subplot(2, 2, 2)
plt.scatter(y_test_orig, y_pred, alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Prediction vs True Values (Test Set)')
plt.text(0.05, 0.9, f'R² = {r2:.3f}', transform=plt.gca().transAxes)

# 3. 残差图
residuals = y_test_orig - y_pred
plt.subplot(2, 2, 3)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.hlines(0, y_pred.min(), y_pred.max(), colors='k', linestyles='dashed')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# 4. 特征重要性 (简单版本)
# 获取第一层权重
weights = model.layers[0].get_weights()[0]
feature_importance = np.sum(np.abs(weights), axis=1)
plt.subplot(2, 2, 4)
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance (First Layer Weights)')

plt.tight_layout()
plt.show()

# # ------------保存模型------------
# model.save('mass_loss_rate_prediction_model.h5')
# print("\n模型已保存为 'mass_loss_rate_prediction_model.h5'")
#
# # 保存标准化器以便后续使用
# import joblib
# joblib.dump(scaler_X, 'scaler_X.save')
# joblib.dump(scaler_y, 'scaler_y.save')
# print("标准化器已保存")