import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 假设我们有3000条历史记录，每条是60分钟的心跳数据（模拟数据）
X_train = np.random.randint(60, 100, (3000, 60))  # 60-100之间的整数模拟心跳
y_train = np.random.randint(70, 90, (3000))       # 模拟未来10分钟平均心跳

# 关键改造点：将输入变为 (样本数, 60分钟, 1个特征)
X_train = X_train.reshape(3000, 60, 1)  # 从 [3000,60] 变为 [3000,60,1]

# 创建模型（对应心跳分析需求）
model = Sequential()

# 第一层卷积：扫描3分钟的心跳窗口，找短期波动
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(60, 1)))
# 想象：16个检查员，每人拿3分钟的窗口，检查心跳是否有异常波动

# 池化层：压缩时间维度，保留关键波动点
model.add(MaxPooling1D(pool_size=2))  # 将60分钟压缩到30个关键时间点

# 第二层卷积：分析更长时间的模式
model.add(Conv1D(32, 3, activation='relu'))  # 32个检查员，分析组合特征

# 展平数据：把时间特征铺平
model.add(Flatten())  # 现在数据变成一维特征向量

# 全连接层：综合所有特征做判断
model.add(Dense(32, activation='relu'))  # 32个神经元分析综合趋势

# 输出层：预测最终心跳数值
model.add(Dense(1))  # 直接输出1个数字（如82）

# 编译模型：用均方误差衡量预测误差
model.compile(optimizer='adam', loss='mse')  # 适合预测具体数值的任务

# 查看模型结构（你会看到每一步的数据形状变化）
model.summary()

# 训练模型：让AI学习心跳规律
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# 使用训练好的模型预测
test_data = np.array([[72,71,75,80,...,85]])  # 实际使用时替换为真实数据
test_data = test_data.reshape(1, 60, 1)      # 格式转换
prediction = model.predict(test_data)
print(f"预测未来10分钟平均心跳：{prediction[0][0]:.1f}次/分钟")