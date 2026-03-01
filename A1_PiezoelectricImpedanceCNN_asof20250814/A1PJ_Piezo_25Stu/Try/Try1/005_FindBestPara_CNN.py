import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import itertools


# ========== 模型组件定义 ==========
def channel_attention(inputs, ratio=8):
    channel = inputs.shape[-1]
    avg_pool = layers.GlobalAveragePooling1D()(inputs)
    max_pool = layers.GlobalMaxPooling1D()(inputs)
    dense1 = layers.Dense(channel // ratio, activation='relu')
    dense2 = layers.Dense(channel, activation='sigmoid')
    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))
    cbam_feature = layers.Multiply()([inputs, layers.Add()([avg_out, max_out])])
    return cbam_feature


def residual_block(x, filters, kernel_size, dropout_rate):
    shortcut = x
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


def build_model(input_shape, filters1, kernel1, filters2, kernel2, dropout_residual, dropout_fc, fc_units, att_ratio):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=filters1, kernel_size=kernel1, activation='relu', padding='same')(inputs)
    x = residual_block(x, filters2, kernel2, dropout_residual)
    x = residual_block(x, filters2, kernel2, dropout_residual)
    x = channel_attention(x, ratio=att_ratio)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_fc)(x)
    x = layers.Dense(fc_units, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# ========== 数据加载与预处理 ==========
def adaptive_smoothing(signal, base_sigma, window):
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(signal, sigma=base_sigma, truncate=window / base_sigma)
    return smoothed


X = np.load('../../Latest1/Imp1_500.npy')
y = np.load('../../Latest1/mass_loss_rate.npy')
X = X.T

# ========== 超参数搜索空间 ==========
param_grid = {
    'base_sigma': [10, 20],
    'window': [30, 50],
    'split_time': [0.7],
    'filters1': [64, 128],
    'kernel1': [5, 7],
    'filters2': [64, 128],
    'kernel2': [3],
    'dropout_residual': [0.2, 0.3],
    'dropout_fc': [0.3],
    'fc_units': [64],
    'att_ratio': [8],
    'epochs': [500],
    'batch_size': [12, 32],
    'learning_rate': [0.001, 0.0005],
    'patience_lr': [3],
    'patience_es': [5],
}

param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

results = []
total_runs = len(param_combinations)

for i, combo in enumerate(param_combinations):
    params = dict(zip(param_names, combo))

    # 检查参数可行性
    skip = False
    # 检查卷积核大小是否大于输入长度
    if params['kernel1'] > X.shape[1] or params['kernel2'] > X.shape[1]:
        print(
            f"Skipping combination {i + 1}: kernel size {params['kernel1']} or {params['kernel2']} too large for input length {X.shape[1]}")
        continue

    print("=" * 40)
    print(f"Running combination {i + 1}/{total_runs}")
    print(', '.join([f"{k}={v}" for k, v in params.items()]))
    print("=" * 40)

    try:
        # 平滑处理
        X_smooth = np.array([adaptive_smoothing(x, params['base_sigma'], params['window']) for x in X])

        # 划分数据
        X_trainval, X_test, y_trainval, y_test = train_test_split(X_smooth, y, test_size=1 - params['split_time'],
                                                                  random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

        # 构建模型
        model = build_model(
            input_shape=(X.shape[1], 1),
            filters1=params['filters1'], kernel1=params['kernel1'],
            filters2=params['filters2'], kernel2=params['kernel2'],
            dropout_residual=params['dropout_residual'],
            dropout_fc=params['dropout_fc'],
            fc_units=params['fc_units'],
            att_ratio=params['att_ratio']
        )

        model.compile(optimizer=optimizers.Adam(learning_rate=params['learning_rate']), loss='mae')

        # 训练
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=params['patience_es'],
                                             restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=params['patience_lr'])

        history = model.fit(
            X_train[..., np.newaxis], y_train,
            validation_data=(X_val[..., np.newaxis], y_val),
            epochs=params['epochs'], batch_size=params['batch_size'], verbose=0,
            callbacks=[early_stop, reduce_lr]
        )

        # 评估
        val_pred = model.predict(X_val[..., np.newaxis]).squeeze()
        test_pred = model.predict(X_test[..., np.newaxis]).squeeze()

        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        abs_errors = np.abs(test_pred - y_test)
        rel_errors = abs_errors / (y_test + 1e-8)

        result = {
            **params,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'best_epoch': len(history.history['loss']),
            'abs_min': np.min(abs_errors),
            'abs_max': np.max(abs_errors),
            'abs_mean': np.mean(abs_errors),
            'rel_min': np.min(rel_errors),
            'rel_max': np.max(rel_errors),
            'rel_mean': np.mean(rel_errors),
        }
        results.append(result)
        pd.DataFrame(results).to_csv('../../Latest1/param_search_results.csv', index=False)

    except Exception as e:
        print(f"Error occurred with combination {i + 1}: {str(e)}")
        continue

# 输出最优组合
if results:  # 只有在有结果时才输出
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['test_mae'].idxmin()]
    print("\nBest Combination:")
    print(best)
else:
    print("No valid combinations were successfully run.")