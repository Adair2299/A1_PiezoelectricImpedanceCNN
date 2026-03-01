import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, GlobalAveragePooling1D, Multiply, Permute, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
matplotlib.use('Qt5Agg')  # 解决plt警告


# ------------识别文件1：质量损失------------
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\有限元\PZT FEM1")
file = "FEM1-3_sasq和质量损失率.xlsx"
file_path = folder / file

if not file_path.exists():
    raise FileNotFoundError(f"文件不存在: {file_path}")

df1 = pd.read_excel(file_path, header=None, skiprows=15, nrows=1, usecols=range(1, 35))
df2 = pd.read_excel(file_path, header=None, skiprows=23, nrows=1, usecols=range(1, 35))
df3 = pd.read_excel(file_path, header=None, skiprows=31, nrows=1, usecols=range(1, 35))
df4 = pd.read_excel(file_path, header=None, skiprows=39, nrows=1, usecols=range(1, 35))

result = pd.concat([df1, df2, df3, df4], ignore_index=True)
result = np.array(result)
result = np.delete(result, 5, axis=1) # 6号钢板数据不要
mass_loss_rate = result.reshape(-1, 1)


# ---------识别文件2：阻抗谱---------
file_path = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\有限元\PZT FEM1\FEM1-Term all 1-500.xlsx")

try:
    df = pd.read_excel(file_path, usecols="B:EG", header=None, skiprows=1, nrows=2496)
    print(f"成功读取，形状为: {df.shape}")  # 应该是 (2496, *)
except Exception as e:
    print(f"读取失败: {e}")

Imp1_500_Orgl = df.values
print(f"最终数组形状: {Imp1_500_Orgl.shape}")  # 应该是(2496, *)

Imp1_500 = np.array(Imp1_500_Orgl)

# 局部方差自适应平滑
from FunA_Adapt_Smoothing import adaptive_smooth
Imp1_500 = adaptive_smooth(Imp1_500, base_sigma=20, window=50, alpha=25)


# --------------------- 打乱前保存原始列编号 --------------------
original_indices = np.arange(1, len(mass_loss_rate)+1)  # 从1开始编号

np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))

mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]
original_indices = original_indices[shuffled_indices]  # 同步打乱


# --------------------- 数据集划分 --------------------
sequence_length = Imp1_500.shape[1]
X = np.array(Imp1_500)
y = np.array(mass_loss_rate)

X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std  # 标准化 X

split_time = int(0.7 * len(X))  # 70%训练
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
idx_train, idx_remaining = original_indices[:split_time], original_indices[split_time:]

X_val, X_test = X_remaining[:len(X_remaining)//2], X_remaining[len(X_remaining)//2:]
y_val, y_test = y_remaining[:len(y_remaining)//2], y_remaining[len(y_remaining)//2:]
idx_val, idx_test = idx_remaining[:len(X_remaining)//2], idx_remaining[len(X_remaining)//2:]

# 3D输入
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)


# --------------------- 模型定义 --------------------
def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])

def TemporalAttention(input_tensor):
    permuted = Permute((2, 1))(input_tensor)
    dense = Dense(input_tensor.shape[1], activation='softmax')(permuted)
    attention = Permute((2, 1))(dense)
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
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)
    x = residual_block(x, 128, 3)
    x = ChannelAttention(x)
    x = TemporalAttention(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model


model = build_advanced_model(sequence_length)
optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
callbacks = [lr_scheduler, early_stop]

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()


# --------------------- 训练 --------------------
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=12,
    validation_data=(X_val, y_val),
    verbose=1
)


# --------------------- 测试 --------------------
y_test_pred = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae = mean_absolute_error(y_test.squeeze(), y_test_pred)
test_absolute_errors = np.abs(y_test.squeeze() - y_test_pred.squeeze())
epsilon = 1e-10
test_relative_errors = np.abs((y_test.squeeze() - y_test_pred.squeeze()) / (y_test.squeeze() + epsilon)) * 100

print(f'''
=== 性能报告 ===

训练样本数：{len(y_train)}，测试样本数：{len(y_test)}
训练集 MAE: {train_mae:.4f}
测试集 MAE: {test_mae:.4f}
误差增加: {((test_mae - train_mae) / train_mae) * 100:.1f}%

绝对误差范围: [{np.min(test_absolute_errors):.4f}, {np.max(test_absolute_errors):.4f}]
平均绝对误差: {np.mean(test_absolute_errors):.4f}

相对误差范围: [{np.min(test_relative_errors):.2f}%, {np.max(test_relative_errors):.2f}%]
平均相对误差: {np.mean(test_relative_errors):.2f}%
''')

# **绘制训练过程**
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.ylim(0,0.006)
plt.pause(0.1)



# --------------------- 绘图 --------------------
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

# 使用等距索引绘图，横坐标标签为原始列编号
plot_idx = np.arange(len(y_test))  # 等距索引
ax1.plot(plot_idx, y_test*100, 'bo-', label='True Value', markersize=4)
ax1.plot(plot_idx, y_test_pred*100, 'rx--', label='Predicted Value', markersize=4)
ax1.set_title('Test Set Prediction Comparison', fontweight='bold')
ax1.set_ylabel('Mass Loss Rate (%)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 25)
ax1.set_xlim(-0.5, len(y_test)-0.5)
ax1.set_xticks(plot_idx)
ax1.set_xticklabels(idx_test)  # 显示原始列编号

ax2.bar(plot_idx, test_absolute_errors*100, color='orange', alpha=0.7, label='Absolute Error')
ax2.set_title('Absolute Errors', fontweight='bold')
ax2.set_xlabel('Original Sample Column')
ax2.set_ylabel('Absolute Error (%)')
ax2.legend()
ax2.grid(True)
ax2.set_xticks(plot_idx)
ax2.set_xticklabels(idx_test)

plt.tight_layout()
plt.show()
