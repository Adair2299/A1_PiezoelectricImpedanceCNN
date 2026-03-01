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
from tensorflow.keras.losses import Huber
matplotlib.use('Qt5Agg')  # 解决plt警告

# 加载数据
Imp1_500orgl = np.load('Imp1_500_RemoveWrongEMI.npy')  # (2496,116)
mass_loss_rateorgl = np.load('mass_loss_rate_RemoveWrongEMI.npy') # (116,1)

# 拆分数据
nsplit = 16
Imp_new = Imp1_500orgl.reshape(int(len(Imp1_500orgl)/nsplit),-1)
mass_new = np.tile(mass_loss_rateorgl, (nsplit, 1))

# 局部方差自适应
from FunA_Adapt_Smoothing import adaptive_smooth
Imp_new = adaptive_smooth(Imp_new, base_sigma=20, window=50, alpha=25)

# --------打乱--------
np.random.seed(2)  # 保证可复现
original_indices = np.arange(len(mass_new))  # 记录原始索引（转置前的列号=mass_new行号）
shuffled_indices = np.random.permutation(len(mass_new))

# 使用相同索引打乱
mass_loss_rate = mass_new[shuffled_indices]
Imp_new = Imp_new.T
Imp1_500 = Imp_new[shuffled_indices]
original_indices_shuffled = original_indices[shuffled_indices]  # 对原始索引也进行相同打乱

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
train_indices = original_indices_shuffled[:split_time]
remaining_indices = original_indices_shuffled[split_time:]

X_val, X_test = X_remaining[:int(len(X_remaining) // 1.1)], X_remaining[int(len(X_remaining) // 1.1):] # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:int(len(y_remaining) // 1.1)], y_remaining[int(len(y_remaining) // 1.1):]
val_indices = remaining_indices[:int(len(X_remaining) // 1.1)]
test_indices = remaining_indices[int(len(X_remaining) // 1.1):]

# --------------------- 第2步：构建模型 --------------------
def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])

def TemporalAttention(input_tensor):
    permuted = Permute((2, 1))(input_tensor)  # [B, C, T]
    dense = Dense(input_tensor.shape[1], activation='softmax', name='temporal_attention')(permuted)
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
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)

    # Residual Blocks
    x = residual_block(x, 128, 3)

    # Attention Mechanisms
    x = ChannelAttention(x)
    x = TemporalAttention(x)

    # Decoder-like Flatten + FC
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    return model

model = build_advanced_model(sequence_length)

# 创建用于获取注意力权重的子模型
attention_model = Model(
    inputs=model.input,
    outputs=model.get_layer('temporal_attention').output
)

# 1. 定义优化器和回调
optimizer = Adam(learning_rate=0.001)  # 初始学习率
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.9, patience=96, min_lr=8.2e-4, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=1000, restore_best_weights=True, verbose=1
)
callbacks = [lr_scheduler, early_stop]  # 组合回调

# 2. 编译模型
model.compile(optimizer=optimizer, loss=Huber(delta=0.02), metrics=['mae'])  # 增加MAE监控
model.summary()

# --------------------- 第3步：训练并监控过拟合 --------------------
history = model.fit(
    X_train, y_train,
    epochs=750, # 迭代次数
    batch_size=128, # 一次用几个训练
    validation_data=(X_val, y_val),
    callbacks = callbacks,
    verbose=1 # 进度条
)

# --------------------- 第4步：测试集评估 --------------------
# 模型预测
y_test_pred = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

# 计算误差
train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae = mean_absolute_error(y_test.squeeze(), y_test_pred)
test_absolute_errors = np.abs(y_test.squeeze() - y_test_pred.squeeze())
epsilon = 1e-10
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



# --------------------- 第5步：时间注意力可视化 --------------------
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10
from datetime import datetime
from matplotlib.gridspec import GridSpec
current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # 格式如：20250808223405
model_folder_name = f"Attention Ablation2-{current_time}"
output_folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\论文材料\消融研究\Attention Ablation") / model_folder_name
output_folder.mkdir(parents=True, exist_ok=True)

# 获取测试集的注意力权重
attention_weights = attention_model.predict(X_test)

# 平均所有样本和通道的注意力权重
avg_attention = np.mean(attention_weights, axis=(0, 1))

# 可视化1: 平均时间注意力权重
plt.figure(figsize=(6, 6))
plt.plot(np.linspace(1, 496, len(avg_attention)), avg_attention, label='Temporal Attention', linewidth=0.5)
plt.title('Average Temporal Attention Weights on Test Set')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Attention Weight')
plt.legend()
plt.grid(True)
plt.pause(0.1)
plt.savefig(output_folder / "Average Temporal Attention Weights on Test Set.png", dpi=600, bbox_inches='tight') # 保存专用
plt.close()


# 可视化2: 单个样本的注意力权重
sample_idx = 0  # 选择第一个测试样本
sample_attention = np.mean(attention_weights[sample_idx], axis=0)  # 平均通道维度

plt.figure(figsize=(6, 6))
plt.plot(np.linspace(1, 496, len(sample_attention)), sample_attention, label=f'Sample {sample_idx} Attention', linewidth=0.5)
plt.title(f'Temporal Attention Weights for Test Sample {sample_idx}')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Attention Weight')
plt.legend()
plt.grid(True)
plt.pause(0.1)
plt.savefig(output_folder / f"Temporal Attention Weights for Test Sample {sample_idx}.png", dpi=600, bbox_inches='tight') # 保存专用
plt.close()



# 可视化3: 原始信号与注意力权重叠加
# 创建画布和子图
for sample_idx in range(48):
    sample_attention = np.mean(attention_weights[sample_idx], axis=0)  # 平均通道维度
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    # 上图：绘制原始信号
    ax1.plot(np.linspace(1, 496, len(X_test[sample_idx])),
            X_test[sample_idx].squeeze(),
            color='tab:blue',
            label='Original Signal', linewidth=0.5)
    ax1.set_ylabel('Normalized Amplitude')
    ax1.set_title(f'Original Signal vs Attention Weights (Sample {sample_idx})')
    ax1.legend()
    ax1.grid(True)
    # 下图：绘制注意力权重（保持0-1范围，不缩放）
    ax2.plot(np.linspace(1, 496, len(sample_attention)),
             sample_attention,
             color='tab:red',
             label='Attention Weights', linewidth=0.5)
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Attention Weight')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(output_folder / f"Original Signal {sample_idx} .png", dpi=600, bbox_inches='tight') # 保存专用
    plt.close()


# 可视化4: 热力图展示多个样本的注意力模式
plt.figure(figsize=(7, 6))
end_idx = 48
plt.imshow(attention_weights[:end_idx].mean(axis=1),  # 移除.T转置
           aspect='auto',
           cmap='viridis',
           extent=[1, 496, 1, end_idx+1])  # x:0-155, y:0-20个样本
plt.colorbar(label='Attention Weight')
plt.xlabel('Frequency (kHz)')  # x轴为时间
plt.ylabel('Sample Index')       # y轴为样本
plt.title(f'Attention Patterns for First {end_idx} Test Samples')
plt.savefig(output_folder / f"Attention Patterns for First {end_idx} Test Samples.png", dpi=600, bbox_inches='tight') # 保存专用
plt.close()
plt.show()