# 一维CNN完整示例：带测试集验证的时序预测（Python 3.10+）
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --------------------- 第1步：生成模拟数据 --------------------
def generate_time_series(size=1000, seq_len=20):
    """生成带噪声的正弦波时序数据"""
    t = np.linspace(0, 10, size)
    data = np.sin(t) + np.random.normal(0, 0.2, size)  # 正弦波+噪声
    return data

# 生成数据并创建输入输出对
data = generate_time_series()
X, y = [], []
sequence_length = 20  # 用过去20个点预测下一个点
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

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

# --------------------- 第4步：测试集评估（关键步骤） --------------------
# 在测试集（模型从未见过的数据）上评估
y_test_pred = model.predict(X_test).flatten()
test_mae = mean_absolute_error(y_test, y_test_pred)

# 对比训练集上的"虚假"表现
y_train_pred = model.predict(X_train).flatten()
train_mae = mean_absolute_error(y_train, y_train_pred)

print(f'''
=== 性能报告 ===
训练集MAE: {train_mae:.4f} （模型见过的数据）
测试集MAE: {test_mae:.4f} （全新数据）
结论：测试集误差比训练集高 {((test_mae - train_mae)/train_mae)*100:.1f}%
''')

# 可视化预测效果
plt.subplot(1,2,2)
plt.plot(y_test[:50], label='True Value')
plt.plot(y_test_pred[:50], label='Predicted Value')
plt.title('Prediction comparison of the first 50 samples in the test set')
plt.legend()
plt.tight_layout()
plt.show()