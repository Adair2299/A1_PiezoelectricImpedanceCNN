# 07月21日，transformer，不好
# 1-500频谱，60天的， 四个周期全部
# 没用6号钢板，只用58组


import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam


# 读取数据
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作")
file = "3_sasq和质量损失率.xlsx"
file_path = folder / file
df1 = pd.read_excel(file_path, header=None, skiprows=15, nrows=1, usecols=range(1, 31))
df2 = pd.read_excel(file_path, header=None, skiprows=23, nrows=1, usecols=range(1, 31))
df3 = pd.read_excel(file_path, header=None, skiprows=31, nrows=1, usecols=range(1, 31))
df4 = pd.read_excel(file_path, header=None, skiprows=39, nrows=1, usecols=range(1, 31))

result = pd.concat([df1, df2, df3, df4], ignore_index=True)
result = np.array(result)
result = np.delete(result, 5, axis=1)  # 去掉6号钢板
mass_loss_rate = result.reshape(-1, 1)

# 读取阻抗数据
file_path_imp = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term all 1-500.xlsx")
df = pd.read_excel(file_path_imp, usecols="B:DM", header=None, skiprows=1, nrows=2496)
Imp1_500_Orgl = df.values
Imp1_500 = np.array(Imp1_500_Orgl)

# 打乱数据
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]

# 数据准备
sequence_length = Imp1_500.shape[1]
X = np.array(Imp1_500)
y = np.array(mass_loss_rate)
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std

# 按时间顺序划分数据集
split_time = int(0.7 * len(X))
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:]
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# Transformer模型



def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=8, key_dim=128)(inputs, inputs)  # 增加头数和维度
    x = LayerNormalization()(x)
    x = Dense(128, activation='relu')(x)  # 增加神经元数
    x = Flatten()(x)
    x = Dropout(0.4)(x)  # 增加dropout防止过拟合
    output = Dense(1)(x)

    model = Model(inputs, output)

    # 使用更小的学习率
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


model = transformer_model((sequence_length, 1))
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=15, batch_size=12, validation_data=(X_val, y_val), verbose=1)

# **绘制训练过程**
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()

# --------------------- 第4步：测试集评估 --------------------

# 模型预测
y_test_pred = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

# 计算 MAE
train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae = mean_absolute_error(y_test.squeeze(), y_test_pred)

# 计算绝对误差（仅测试集）
test_absolute_errors = np.abs(y_test.squeeze() - y_test_pred.squeeze())

# 计算相对误差（避免除以0，加一个小的epsilon）
epsilon = 1e-6
test_relative_errors = np.abs((y_test.squeeze() - y_test_pred.squeeze()) / (y_test.squeeze() + epsilon)) * 100

# --------------------- 打印性能报告 --------------------

print(f'''
=== 性能报告 ===

训练样本数：{len(y_train)}，测试样本数：{len(y_test)}
训练集 MAE: {train_mae:.4f}
测试集 MAE: {test_mae:.4f}
误差增加: {((test_mae - train_mae) / train_mae) * 100:.1f}%

--- 测试集详细误差分析 ---

绝对误差范围: [{np.min(test_absolute_errors):.4f}, {np.max(test_absolute_errors):.4f}]
平均绝对误差: {np.mean(test_absolute_errors):.4f}

相对误差范围: [{np.min(test_relative_errors):.2f}%, {np.max(test_relative_errors):.2f}%]
平均相对误差: {np.mean(test_relative_errors):.2f}%
''')

# **绘制真实值 vs. 预测值**
plt.figure(figsize=(5, 5))
plt.plot(y_test, 'bo-', label='True Value')
plt.plot(y_test_pred, 'rx--', label='Predicted Value')
plt.title(f'Test Set Prediction Comparison ({len(y_test)} Samples)')
plt.legend()
plt.grid(True)
plt.ylim(-0.15, 0.3)
plt.tight_layout()
plt.show()
