# 04月27日可行,06月07日可行，基于004普通神经网络的改进预测massloss
# 1-500频谱，60天的， 四个周期全部
# 没用6号钢板，只用58组



import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # 解决plt警告
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, GlobalAveragePooling1D, Multiply, Permute, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf



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
mass_loss_rate = result.reshape(-1, 1)


# ---------识别文件2：阻抗谱 Piezoelectric Impedance Spectrum---------

# 定义文件路径


# 文件路径
file_path = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term all 1-500.xlsx")

# 精确读取 B2:DM2497
try:
    df = pd.read_excel(file_path, usecols="B:DM", header=None, skiprows=1, nrows=2496)
    print(f"成功读取，形状为: {df.shape}")  # 应该是 (2496, 117)

except Exception as e:
    print(f"读取失败: {e}")

Imp1_500_Orgl = df.values
print(f"最终数组形状: {Imp1_500_Orgl.shape}")  # 应该是(116, 2496)

Imp1_500 = np.array(Imp1_500_Orgl)

# --------打乱--------
# 生成相同的随机排列索引
np.random.seed(2)  # 可选：设置随机种子（保证结果可复现）
shuffled_indices = np.random.permutation(len(mass_loss_rate))

# 使用相同的索引打乱两个矩阵
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]



##########################################################
# --------------------- 第1步：整理数据 -------------------


sequence_length = Imp1_500.shape[1]  # 用2496个点预测mass_loss

X = np.array(Imp1_500)  # 阻抗数据
y = np.array(mass_loss_rate)  # 质量损失

# **修改点：仅标准化 X，y 直接保持物理值**
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std  # 标准化 X
# y 保持原始尺度 (y 的单位是质量损失，不能归一化)

# **严格按时间顺序划分数据集**
split_time = int(0.7 * len(X))  # 70% 训练
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:] # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

# **转换为3D格式** [样本数, 时间步, 特征]
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# --------------------- 第2步：构建容易过拟合的模型 --------------------
# model = Sequential([
#     Conv1D(32, 4, activation='relu', input_shape=(sequence_length, 1)),
#     Conv1D(16, 2, activation='relu'),
#     Flatten(),
#     Dense(8, activation='relu'),
#     Dense(1)  # 直接预测质量损失
# ])



def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])

def TemporalAttention(input_tensor):
    permuted = Permute((2, 1))(input_tensor)  # [B, C, T]
    dense = Dense(input_tensor.shape[1], activation='softmax')(permuted)
    attention = Permute((2, 1))(dense)  # [B, T, C]
    return Multiply()([input_tensor, attention])

def residual_block(x, filters, kernel_size, pooling=True):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Add()([shortcut, x])
    x = LayerNormalization()(x)
    if pooling:
        x = MaxPooling1D(pool_size=2)(x)
    return x

def build_advanced_model(input_length):
    inputs = Input(shape=(input_length, 1))

    # Initial Conv
    x = Conv1D(64, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)

    # Residual Blocks
    x = residual_block(x, 64, 3)
    x = residual_block(x, 64, 3)

    # Attention Mechanisms
    x = ChannelAttention(x)
    x = TemporalAttention(x)

    # Decoder-like Flatten + FC
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    return model


model = build_advanced_model(sequence_length)
model.compile(optimizer='adam', loss='mse')
model.summary()


# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=100, # 迭代次数
    batch_size=12, # 一次用几个训练
    validation_data=(X_val, y_val),
    verbose=1 # 进度条
)

# **绘制训练过程**
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.pause(0.1)


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



# # 打印真实值与预测值的对比（测试集）
# print("=== 测试集真实值 vs. 预测值 ===")
# for i, (true_val, pred_val) in enumerate(zip(y_test.squeeze(), y_test_pred.squeeze())):
#     abs_err = abs(true_val - pred_val)
#     print(f"样本{i:2d}: 真实值={true_val:.4f}, 预测值={pred_val:.4f}, 绝对误差={abs_err:.4f}")
#
#


# **绘制真实值 vs. 预测值**
plt.figure(figsize=(5, 5))
plt.plot(y_test, 'bo-', label='True Value')
plt.plot(y_test_pred, 'rx--', label='Predicted Value')
plt.title(f'Test Set Prediction Comparison ({len(y_test)} Samples)')
plt.legend()
plt.grid(True)

plt.ylim(-0.15, 0.3)
plt.show()

