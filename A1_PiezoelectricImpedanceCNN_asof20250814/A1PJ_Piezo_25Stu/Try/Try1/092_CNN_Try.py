# 一维CNN完整示例：用历史股价预测未来价格
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# --------------------- 第1步：生成模拟真实数据 --------------------
# 生成100天的股票历史数据（100个样本，每个样本是连续5天的股价）
num_samples = 1000
sequence_length = 5  # 用过去5天的价格预测第6天

# 模拟股价数据：初始价格100，每天随机波动±2%
np.random.seed(42)  # 固定随机种子确保可重复性
prices = [100.0]
for _ in range(num_samples + sequence_length):
    change = np.random.uniform(-0.02, 0.02)  # -2%到+2%的随机波动
    prices.append(prices[-1] * (1 + change))

# 创建输入输出对（滑动窗口）
X, y = [], []
for i in range(len(prices) - sequence_length):
    X.append(prices[i:i+sequence_length])  # 输入：连续5天价格
    y.append(prices[i+sequence_length])    # 输出：第6天价格

X = np.array(X)
y = np.array(y)

# 数据标准化（提升模型性能）
mean, std = X.mean(), X.std()
X = (X - mean) / std
y = (y - mean) / std

# 转换为三维输入 [样本数, 时间步长, 特征数]
X = X.reshape(X.shape[0], sequence_length, 1)

# 分割训练集/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------- 第2步：构建1D CNN模型 --------------------
model = Sequential([
    Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(sequence_length, 1)),
    MaxPooling1D(pool_size=1),  # 由于序列短，池化窗口设为1
    Flatten(),
    Dense(8, activation='relu'),
    Dense(1)  # 输出标准化后的价格
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# --------------------- 第3步：训练模型 --------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# --------------------- 第4步：预测示例 --------------------
# 生成测试输入（需要是5天的标准化数据）
# test_input = np.array([120, 118, 122, 125, 123])  # 假设近5天实际价格
test_input = np.array([100.        ,  99.49816048, 101.29197025, 102.23193519,
       102.6353771])  # 假设近5天实际价格
test_input = (test_input - mean) / std  # 必须使用相同的标准化参数
test_input = test_input.reshape(1, sequence_length, 1)  # 转换为三维

# 进行预测并反标准化
predicted_normalized = model.predict(test_input)[0][0]
predicted_price = predicted_normalized * std + mean

print(f"\n预测结果：")
print(f"输入5天价格：{[round(x,1) for x in test_input.reshape(-1) * std + mean]}")
print(f"预测第6天价格：{predicted_price:.1f}")