import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, \
    GlobalAveragePooling1D, Multiply, Permute, Reshape, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
from tensorflow.keras.losses import Huber

matplotlib.use('Qt5Agg')  # 解决plt警告

# 加载数据
Imp1_500orgl = np.load('../../Latest2/Ablation/Imp1_500_RemoveWrongEMI.npy')  # (2496,116)
mass_loss_rateorgl = np.load('../../Latest2/Ablation/mass_loss_rate_RemoveWrongEMI.npy')  # (116,1)

# 拆分数据
nsplit = 16
Imp_new = Imp1_500orgl.reshape(int(len(Imp1_500orgl) / nsplit), -1)
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

X_val, X_test = X_remaining[:int(len(X_remaining) // 1.1)], X_remaining[int(len(X_remaining) // 1.1):]  # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:int(len(y_remaining) // 1.1)], y_remaining[int(len(y_remaining) // 1.1):]
val_indices = remaining_indices[:int(len(X_remaining) // 1.1)]
test_indices = remaining_indices[int(len(X_remaining) // 1.1):]


# --------------------- 第2步：构建模型（修改以提取注意力权重）-------------------
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
    # 返回加权结果和注意力权重
    return Multiply()([input_tensor, attention]), attention


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

    # 修改：捕获注意力权重
    x, temporal_weights = TemporalAttention(x)  # 返回处理结果和权重

    # Decoder-like Flatten + FC
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    # 创建同时输出预测值和注意力权重的模型
    model = Model(inputs, [outputs, temporal_weights])
    return model


model = build_advanced_model(sequence_length)

# 1. 定义优化器和回调
optimizer = Adam(learning_rate=0.001)  # 初始学习率
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.9, patience=96, min_lr=8.2e-4, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=1000, restore_best_weights=True, verbose=1
)
callbacks = [lr_scheduler, early_stop]  # 组合回调

# 2. 编译模型（注意：现在有两个输出）
model.compile(
    optimizer=optimizer,
    loss=[Huber(delta=0.02), None],  # 只计算第一个输出（预测值）的损失
    metrics={'output_1': 'mae'}  # 只监控第一个输出的MAE
)
model.summary()

# --------------------- 第3步：训练模型 --------------------
history = model.fit(
    X_train, [y_train, np.zeros_like(y_train)],  # 为第二个输出提供虚拟标签
    epochs=5,
    batch_size=128,
    validation_data=(X_val, [y_val, np.zeros_like(y_val)]),
    callbacks=callbacks,
    verbose=1
)

# --------------------- 第4步：测试集评估 --------------------

# 模型预测（现在返回预测值和注意力权重）
y_test_pred, test_attentions = model.predict(X_test)
y_train_pred, train_attentions = model.predict(X_train)

# 计算误差
y_test_pred = y_test_pred.squeeze()
y_train_pred = y_train_pred.squeeze()

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


# --------------------- 第5步：可视化指定样本的注意力权重 --------------------
def visualize_attention_for_original_sample(original_index, Imp_new, all_indices, X_set, attention_set, set_name):
    """
    可视化原始样本的注意力权重

    参数:
    original_index: 在原始未打乱数据中的索引
    Imp_new: 原始未打乱数据 (156, 1728)
    all_indices: 当前数据集对应的原始索引数组
    X_set: 标准化后的输入数据
    attention_set: 注意力权重数组
    set_name: 数据集名称 ('train', 'val', 'test')
    """
    # 在指定数据集中查找样本
    try:
        # 在索引数组中查找位置
        pos_in_set = np.where(all_indices == original_index)[0][0]
        print(f"样本 {original_index} 在{set_name}集中的位置: {pos_in_set}")
    except IndexError:
        print(f"错误: 索引 {original_index} 不在{set_name}集中")
        return

    # 获取原始信号 (反转标准化)
    original_signal = Imp_new[:, original_index]

    # 获取注意力权重 (在通道维度上取平均)
    attention_weights = np.mean(attention_set[pos_in_set], axis=-1).squeeze()

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # 绘制原始信号
    color = 'tab:blue'
    ax1.set_xlabel('数据点索引')
    ax1.set_ylabel('阻抗值', color=color)
    ax1.plot(original_signal, color=color, label='原始信号')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个y轴用于注意力权重
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('注意力权重', color=color)
    ax2.fill_between(range(len(attention_weights)), 0, attention_weights,
                     color=color, alpha=0.3, label='注意力权重')
    ax2.plot(attention_weights, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加重要点标记
    top5_indices = np.argsort(attention_weights)[-5:][::-1]
    for idx in top5_indices:
        ax1.plot(idx, original_signal[idx], 'go', markersize=8)
        ax2.plot(idx, attention_weights[idx], 'ro', markersize=8)

    plt.title(f'原始索引 {original_index} 的注意力权重可视化 ({set_name}集)')
    fig.tight_layout()
    plt.legend()
    plt.show()

    # 返回关键信息
    return {
        'original_index': original_index,
        'position_in_set': pos_in_set,
        'signal': original_signal,
        'attention_weights': attention_weights,
        'top5_points': [(i, original_signal[i], attention_weights[i]) for i in top5_indices]
    }


# 使用示例：可视化原始索引为10的样本（在测试集中）
# 注意：确保该索引存在于测试集中
target_original_index = 10  # 更改为你想可视化的原始索引

# 在测试集中查找
if target_original_index in test_indices:
    result = visualize_attention_for_original_sample(
        target_original_index,
        Imp_new,
        test_indices,
        X_test,
        test_attentions,
        'test'
    )
    print("\n注意力最高的5个点:")
    for idx, sig_val, attn_val in result['top5_points']:
        print(f"数据点 {idx}: 信号值={sig_val:.4f}, 注意力权重={attn_val:.4f}")

# 在训练集中查找
elif target_original_index in train_indices:
    result = visualize_attention_for_original_sample(
        target_original_index,
        Imp_new,
        train_indices,
        X_train,
        train_attentions,
        'train'
    )
    print("\n注意力最高的5个点:")
    for idx, sig_val, attn_val in result['top5_points']:
        print(f"数据点 {idx}: 信号值={sig_val:.4f}, 注意力权重={attn_val:.4f}")

# 在验证集中查找
elif target_original_index in val_indices:
    result = visualize_attention_for_original_sample(
        target_original_index,
        Imp_new,
        val_indices,
        X_val,
        model.predict(X_val)[1],  # 预测验证集注意力
        'val'
    )
    print("\n注意力最高的5个点:")
    for idx, sig_val, attn_val in result['top5_points']:
        print(f"数据点 {idx}: 信号值={sig_val:.4f}, 注意力权重={attn_val:.4f}")

else:
    print(f"错误: 索引 {target_original_index} 在任何数据集中都未找到")