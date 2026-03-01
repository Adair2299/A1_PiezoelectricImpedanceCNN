import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, \
    GlobalAveragePooling1D, Multiply, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm.keras import TqdmCallback
import optuna
from optuna.visualization import plot_optimization_history

# 加载数据
Imp1_500 = np.load('../Latest1/Imp1_500.npy')  # (2496,116)
mass_loss_rate = np.load('../Latest1/mass_loss_rate.npy')

# 局部方差自适应
from FunA_Adapt_Smoothing import adaptive_smooth

Imp1_500 = adaptive_smooth(Imp1_500, base_sigma=20, window=50, alpha=25)

# --------打乱--------
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]

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
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:]  # 15% 验证, 15% 测试
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

# **转换为3D格式** [样本数, 时间步, 特征]
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)


# --------------------- 定义模型 ---------------------

# 残差块定义 (新增参数：是否使用LayerNorm和Dropout)
def residual_block(x, filters, kernel_size, pooling=True, pool_size=2, use_ln=True, use_dropout=True):
    shortcut = x
    # 如果维度不匹配，使用1x1卷积调整shortcut
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    if use_ln:
        x = LayerNormalization()(x)
    if use_dropout:
        x = Dropout(0.2)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Add()([shortcut, x])
    if use_ln:
        x = LayerNormalization()(x)
    if pooling:
        # 只有在特征图尺寸足够大时才进行池化
        if x.shape[1] // pool_size >= 1:
            x = MaxPooling1D(pool_size=pool_size)(x)
    return x


# 通道注意力机制
def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])


# 时序注意力机制
def TemporalAttention(input_tensor):
    permuted = Permute((2, 1))(input_tensor)  # [B, C, T]
    dense = Dense(input_tensor.shape[1], activation='softmax')(permuted)
    attention = Permute((2, 1))(dense)  # [B, T, C]
    return Multiply()([input_tensor, attention])


# 构建模型 (新增残差块配置参数)
def build_advanced_model(input_length, filters, kernel_size, dropout_rate,
                         n_residual_blocks, initial_pool_size,
                         residual_pool_size, use_ln, use_dropout):
    inputs = Input(shape=(input_length, 1))

    # Initial Conv
    x = Conv1D(filters, kernel_size, activation='relu', padding='same')(inputs)

    # 初始池化 - 确保不会过度压缩
    if input_length // initial_pool_size >= 1:
        x = MaxPooling1D(pool_size=initial_pool_size)(x)
    else:
        initial_pool_size = 1  # 不进行池化

    # Residual Blocks with configurable components
    for i in range(n_residual_blocks):
        # 只在特征图尺寸足够大时才进行池化
        pooling = (i < n_residual_blocks - 1) and (x.shape[1] // residual_pool_size >= 1)
        x = residual_block(x, filters, kernel_size,
                           pooling=pooling,
                           pool_size=residual_pool_size,
                           use_ln=use_ln,
                           use_dropout=use_dropout)

    # Attention Mechanisms
    x = ChannelAttention(x)
    x = TemporalAttention(x)

    # Decoder-like Flatten + FC
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    return model


# 目标函数，传入超参数进行训练
def objective(trial):
    # 固定其他参数，只优化batch_size和validation_split
    filters = 64
    learning_rate = 1e-3
    dropout_rate = 0.3
    kernel_size = 7
    n_residual_blocks = 3
    initial_pool_size = 4
    residual_pool_size = 2
    use_ln = True
    use_dropout = True

    # 需要优化的参数
    batch_size = trial.suggest_categorical('batch_size', [8, 12, 16, 24, 32])
    val_split = trial.suggest_float('val_split', 0.1, 0.3)  # 验证集比例

    try:
        # 构建模型
        model = build_advanced_model(
            input_length=Imp1_500.shape[1],
            filters=filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            n_residual_blocks=n_residual_blocks,
            initial_pool_size=initial_pool_size,
            residual_pool_size=residual_pool_size,
            use_ln=use_ln,
            use_dropout=use_dropout
        )

        # 编译模型
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

        # 训练模型 - 使用validation_split代替固定验证集
        history = model.fit(
            X_train, y_train,
            epochs=150,  # 减少epochs以加快优化
            batch_size=batch_size,
            validation_split=val_split,  # 使用动态验证集比例
            verbose=0
        )

        # 获取验证集的MAE
        val_mae = history.history['val_mae'][-1]
        print(f"Validation MAE for trial: {val_mae:.4f} (batch_size={batch_size}, val_split={val_split:.2f})")

        return val_mae
    except Exception as e:
        print(f"Error in trial: {e}")
        return float('inf')  # 返回一个很大的值，表示这个配置不好


# 进行贝叶斯优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

plot_optimization_history(study).show()

# 打印优化结果
print(f"Best trial: {study.best_trial.params}")
best_params = study.best_trial.params

# 保存最佳结果到 CSV 文件
best_results = pd.DataFrame([best_params])
best_results.to_csv("best_hyperparameters_batch_val.csv", index=False)

# 输出最终的性能报告
print(f"Best validation MAE: {study.best_value:.4f}")