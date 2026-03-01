import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, MaxPooling1D, LayerNormalization, Add, \
    GlobalAveragePooling1D, Multiply, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import optuna
from optuna.visualization import plot_optimization_history

# --------------------- 数据准备 ---------------------
Imp1_500 = np.load('../Latest1/Imp1_500.npy')  # (2496,116)
mass_loss_rate = np.load('../Latest1/mass_loss_rate.npy')

# 局部方差自适应
from FunA_Adapt_Smoothing import adaptive_smooth

Imp1_500 = adaptive_smooth(Imp1_500, base_sigma=20, window=50, alpha=25)

# 数据划分和标准化
np.random.seed(2)
shuffled_indices = np.random.permutation(len(mass_loss_rate))
mass_loss_rate = mass_loss_rate[shuffled_indices]
Imp1_500 = Imp1_500[shuffled_indices]

sequence_length = Imp1_500.shape[1]
X = (Imp1_500 - Imp1_500.mean()) / Imp1_500.std()
y = mass_loss_rate

split_time = int(0.7 * len(X))
X_train, X_remaining = X[:split_time], X[split_time:]
y_train, y_remaining = y[:split_time], y[split_time:]
X_val, X_test = X_remaining[:len(X_remaining) // 2], X_remaining[len(X_remaining) // 2:]
y_val, y_test = y_remaining[:len(y_remaining) // 2], y_remaining[len(y_remaining) // 2:]

X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)


# --------------------- 定义模型组件 ---------------------
def residual_block(x, filters, kernel_size, pooling=True, pool_size=2):
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Add()([shortcut, x])
    x = LayerNormalization()(x)
    if pooling and x.shape[1] // pool_size >= 1:
        x = MaxPooling1D(pool_size=pool_size)(x)
    return x


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


# --------------------- 模型构建函数 ---------------------
def build_advanced_model(input_length, dense_config):
    inputs = Input(shape=(input_length, 1))

    # 初始卷积和残差块
    x = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, 128, 3)
    x = ChannelAttention(x)
    x = TemporalAttention(x)
    x = GlobalAveragePooling1D()(x)

    # 动态构建Dense层
    for units, dropout_rate in dense_config:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(1)(x)
    return Model(inputs, outputs)


# --------------------- 贝叶斯优化 ---------------------
def objective(trial):
    # 1. 确定Dense层数量 (1-3层)
    n_layers = trial.suggest_int('n_layers', 1, 3)

    # 2. 为每一层定义独立的单元数和dropout率
    dense_config = []
    for i in range(n_layers):
        units = trial.suggest_categorical(f'units_{i}', [32, 64, 128, 256])
        dropout = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        dense_config.append((units, dropout))

    try:
        model = build_advanced_model(
            input_length=sequence_length,
            dense_config=dense_config
        )

        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=12,
            verbose=0
        )

        return history.history['val_mae'][-1]
    except Exception as e:
        print(f"Error in trial: {e}")
        return float('inf')


# --------------------- 运行优化 ---------------------
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# 打印最佳配置
print("Best Dense Layer Configuration:")
best_params = study.best_params
n_layers = best_params['n_layers']
for i in range(n_layers):
    print(f"  Layer {i + 1}: Units={best_params[f'units_{i}']}, Dropout={best_params[f'dropout_{i}']:.2f}")

# 保存最佳配置
pd.DataFrame([best_params]).to_csv("best_dense_config.csv", index=False)

# --------------------- 训练最终模型 ---------------------
print("\nTraining final model...")
final_config = []
for i in range(best_params['n_layers']):
    final_config.append((
        best_params[f'units_{i}'],
        best_params[f'dropout_{i}']
    ))

final_model = build_advanced_model(sequence_length, final_config)
final_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

history = final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=350,
    batch_size=12,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
)

# 评估测试集
test_mae = final_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\nTest MAE: {test_mae:.4f}")