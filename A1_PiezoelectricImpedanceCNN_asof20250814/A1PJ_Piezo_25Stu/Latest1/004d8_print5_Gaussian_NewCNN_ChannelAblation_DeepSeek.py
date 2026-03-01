import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, \
    GlobalAveragePooling1D, Multiply, Permute, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

matplotlib.use('Qt5Agg')  # 解决plt警告

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10


# --------------------- 自定义层定义 --------------------
def ChannelAttention(input_tensor, reduction_ratio=4):
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


# --------------------- 数据加载和预处理 --------------------
# 加载数据
Imp1_500 = np.load('../Latest1/Imp1_500.npy')  # (2496,116)
mass_loss_rate = np.load('../Latest1/mass_loss_rate.npy')

# 打乱前保存原始数据和索引
original_indices = np.arange(len(Imp1_500))
Imp1_500_before_shuffle = Imp1_500.copy()
mass_loss_rate_before_shuffle = mass_loss_rate.copy()

# 打乱数据
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))
Imp1_500 = Imp1_500[shuffled_indices]
mass_loss_rate = mass_loss_rate[shuffled_indices]

# 创建索引映射字典
index_mapping = {}
for new_idx, original_idx in enumerate(shuffled_indices):
    index_mapping[original_idx] = new_idx

# 整理数据
sequence_length = Imp1_500.shape[1]
X = np.array(Imp1_500)
y = np.array(mass_loss_rate)

# 标准化
X_mean, X_std = X.mean(), X.std()
X_normalized = (X - X_mean) / X_std

# 划分数据集
split_time = int(0.7 * len(X))
X_train, X_remaining = X_normalized[:split_time], X_normalized[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:]
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

# 转换为3D格式
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# 记录测试集原始索引
test_original_indices = shuffled_indices[split_time + len(X_val):]

# --------------------- 模型训练 --------------------
model = build_advanced_model(sequence_length)
optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
callbacks = [lr_scheduler, early_stop]

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=12,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# --------------------- 模型评估 --------------------
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

--- 测试集详细误差分析 ---

绝对误差范围: [{np.min(test_absolute_errors):.4f}, {np.max(test_absolute_errors):.4f}]
平均绝对误差: {np.mean(test_absolute_errors):.4f}

相对误差范围: [{np.min(test_relative_errors):.2f}%, {np.max(test_relative_errors):.2f}%]
平均相对误差: {np.mean(test_relative_errors):.2f}%
''')


# --------------------- 可视化函数 --------------------
def plot_grad_cam(original_idx):
    """可视化特定原始索引样本的频率贡献"""
    # 查找样本在测试集中的位置
    if original_idx not in test_original_indices:
        print(f"原始索引 {original_idx} 的样本不在测试集中")
        return

    # 获取标准化样本
    sample_idx_in_test = np.where(test_original_indices == original_idx)[0][0]
    sample_normalized = X_test[sample_idx_in_test]
    sample_expanded = np.expand_dims(sample_normalized, axis=0)

    # 获取原始信号（未标准化）
    raw_signal = Imp1_500_before_shuffle[original_idx]

    # 找到最后一个卷积层
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv1D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        print("未找到卷积层")
        return

    # 创建Grad-CAM模型
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # 计算梯度
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(sample_expanded)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.reduce_sum(conv_outputs * weights, axis=-1)
    cam = tf.squeeze(cam).numpy()

    # 应用ReLU和归一化
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)

    # 插值到原始长度
    length_before_pooling = conv_outputs.shape[1]
    x_original = np.arange(sequence_length)
    x_cam = np.linspace(0, sequence_length - 1, num=length_before_pooling)
    f = interp1d(x_cam, cam, kind='linear', fill_value="extrapolate")
    cam_resized = f(x_original)

    # 创建频率轴 (1-500 kHz)
    freq = np.linspace(1, 500, sequence_length)

    # 创建绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 子图1: 原始信号和贡献热力图
    im = ax1.scatter(freq, raw_signal, c=cam_resized, cmap='jet', s=15,
                     norm=mcolors.Normalize(vmin=0, vmax=1))
    ax1.plot(freq, raw_signal, 'k-', alpha=0.3)
    ax1.set_ylabel('Impedance (Ω)')
    ax1.set_title(f'原始信号和频率贡献 (样本原始索引: {original_idx})')
    fig.colorbar(im, ax=ax1, label='贡献强度')

    # 子图2: 贡献分布
    ax2.plot(freq, cam_resized, 'r-', linewidth=1.5)
    ax2.fill_between(freq, 0, cam_resized, color='red', alpha=0.3)
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('标准化贡献')
    ax2.set_title('频率贡献分布')
    ax2.set_xlim(1, 500)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # 打印预测信息
    prediction = model.predict(sample_expanded)[0][0]
    true_value = mass_loss_rate_before_shuffle[original_idx][0]
    error = abs(true_value - prediction)
    rel_error = (error / true_value) * 100

    print(f"\n样本原始索引: {original_idx}")
    print(f"真实质量损失率: {true_value * 100:.4f}%")
    print(f"预测质量损失率: {prediction * 100:.4f}%")
    print(f"绝对误差: {error * 100:.4f}%")
    print(f"相对误差: {rel_error:.2f}%")

    return cam_resized


# --------------------- 主程序 --------------------
if __name__ == "__main__":
    # 示例: 可视化第一个测试样本
    plot_grad_cam(test_original_indices[0])

    # 用户输入可视化样本
    while True:
        try:
            user_input = input("\n输入要可视化的原始索引(0-{})或q退出: ".format(len(Imp1_500_before_shuffle) - 1))
            if user_input.lower() == 'q':
                break
            orig_idx = int(user_input)
            if orig_idx < 0 or orig_idx >= len(Imp1_500_before_shuffle):
                print("索引超出范围")
            else:
                plot_grad_cam(orig_idx)
        except ValueError:
            print("请输入有效数字")