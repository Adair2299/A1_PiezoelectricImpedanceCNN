import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LayerNormalization, Add, \
    GlobalAveragePooling1D, Multiply, Permute
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.interpolate import interp1d
from tensorflow.keras.losses import Huber
matplotlib.use('Qt5Agg')  # 解决plt警告

# 加载数据
Imp1_500orgl = np.load('../../Latest2/Ablation/Imp1_500_RemoveWrongEMI.npy')  # (2496,116)
mass_loss_rateorgl = np.load('../../Latest2/Ablation/mass_loss_rate_RemoveWrongEMI.npy') # (116,1)

# 拆分数据
nsplit = 16
Imp_new = Imp1_500orgl.reshape(int(len(Imp1_500orgl)/nsplit), -1)
mass_new = np.tile(mass_loss_rateorgl, (nsplit, 1))

# 局部方差自适应
from FunA_Adapt_Smoothing import adaptive_smooth
Imp_new = adaptive_smooth(Imp_new, base_sigma=20, window=50, alpha=25)

# 打乱数据
np.random.seed(2)
original_indices = np.arange(len(mass_new))
shuffled_indices = np.random.permutation(len(mass_new))
mass_loss_rate = mass_new[shuffled_indices]
Imp_new = Imp_new.T
Imp1_500 = Imp_new[shuffled_indices]
original_indices_shuffled = original_indices[shuffled_indices]

# 数据整理
sequence_length = Imp1_500.shape[1]
X = np.array(Imp1_500)
y = np.array(mass_loss_rate)
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std

split_time = int(0.7 * len(X))
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
train_indices = original_indices_shuffled[:split_time]
remaining_indices = original_indices_shuffled[split_time:]

X_val, X_test = X_remaining[:int(len(X_remaining) // 1.1)], X_remaining[int(len(X_remaining) // 1.1):]
y_val, y_test = y_remaining[:int(len(y_remaining) // 1.1)], y_remaining[int(len(y_remaining) // 1.1):]
val_indices = remaining_indices[:int(len(X_remaining) // 1.1)]
test_indices = remaining_indices[int(len(X_remaining) // 1.1):]

# 模型组件
def ChannelAttention(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    dense = Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    dense = Dense(channel, activation='sigmoid')(dense)
    return Multiply()([input_tensor, dense])

def TemporalAttentionWithWeights(input_tensor):
    permuted = Permute((2, 1))(input_tensor)  # [B, C, T]
    dense = Dense(input_tensor.shape[1], activation='softmax')(permuted)
    attention = Permute((2, 1))(dense)  # [B, T, C]
    output = Multiply()([input_tensor, attention])
    return output, attention

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

def build_advanced_model_with_att(input_length):
    inputs = Input(shape=(input_length, 1))
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=3)(x)
    x = residual_block(x, 128, 3)
    x = ChannelAttention(x)
    x, att_weights = TemporalAttentionWithWeights(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, name='pred_output')(x)
    model = Model(inputs, [outputs, att_weights])
    return model

# 构建并编译
model = build_advanced_model_with_att(sequence_length)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss={'pred_output': Huber(delta=0.02), 'permute_2': lambda y_true, y_pred: 0.0},
              metrics={'pred_output': 'mae'})

# 训练
history = model.fit(
    X_train[..., np.newaxis],
    {'pred_output': y_train, 'permute_2': np.zeros((len(y_train), sequence_length//6, 128))},
    epochs=5,
    batch_size=128,
    validation_data=(X_val[..., np.newaxis],
                     {'pred_output': y_val, 'permute_2': np.zeros((len(y_val), sequence_length//6, 128))}),
    verbose=1
)

# 预测并提取注意力
y_pred_test, att_test = model.predict(X_test[..., np.newaxis])
att_test_mean = np.mean(att_test, axis=-1)  # [样本数, 时间步]

# 插值到 2496 点
pooled_length = att_test_mean.shape[1]
time_original = np.linspace(0, sequence_length-1, sequence_length)
time_pooled = np.linspace(0, sequence_length-1, pooled_length)
interp_func = interp1d(time_pooled, att_test_mean[0], kind='linear')
att_resized = interp_func(time_original)

# 可视化
plt.figure(figsize=(12,4))
plt.plot(time_original, att_resized, label="Temporal Attention (Interpolated to 2496)")
plt.xlabel("Time Step (Original 2496)")
plt.ylabel("Attention Weight")
plt.title("Temporal Attention Visualization - Test Sample 0")
plt.legend()
plt.show()
