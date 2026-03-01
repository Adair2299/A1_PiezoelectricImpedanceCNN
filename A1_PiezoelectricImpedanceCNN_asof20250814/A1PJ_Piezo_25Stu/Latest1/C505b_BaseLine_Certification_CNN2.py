# -*- coding: utf-8 -*-
# 神经网络预测 mass loss (使用基线方程参数)
# 2025‑07‑04

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from sklearn.metrics import mean_absolute_error

# ----------- 第1步：加载质量损失 -----------
folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作")
loss_file = folder / "3_sasq和质量损失率.xlsx"
rows = [15, 23, 31, 39]                     # B16, B24, B32, B40
dfs = [pd.read_excel(loss_file, header=None, skiprows=r,
                     nrows=1, usecols=range(1, 31)) for r in rows]
mass_loss = pd.concat(dfs, ignore_index=True).to_numpy()
mass_loss = np.delete(mass_loss, 5, axis=1)           # 删除 6 号钢板
y = mass_loss.reshape(-1, 1)                          # 共 116 组

# ----------- 第2步：加载基线参数 -----------
para_path = folder / "10_1_基线假设方程全部参数汇总1.xlsx"
sheets = [pd.read_excel(para_path, sheet_name=f"Term{i}", header=None,
                        skiprows=1, nrows=14, usecols="B:AD") for i in range(1, 5)]
X = pd.concat(sheets, axis=1).values.T                # shape: (116, 14)

# ----------- 第3步：数据预处理 -----------
X = (X - X.mean(axis=0)) / X.std(axis=0)              # 按列标准化

# ==========================================================
# 方法 1：随机打乱再 7:1.5:1.5 划分  ------------------------
# ==========================================================
# np.random.seed(2)
# shuffle_idx = np.random.permutation(len(X))           # 打乱索引
# orig_idx_shuffled = shuffle_idx.copy()                # ****** 新增 ******
#
# X, y = X[shuffle_idx], y[shuffle_idx]
#
# split1 = int(0.7 * len(X))
# split2 = split1 + (len(X) - split1) // 2
#
# train_idx = orig_idx_shuffled[:split1]                # ****** 新增 ******
# val_idx   = orig_idx_shuffled[split1:split2]          # ****** 新增 ******
# test_idx  = orig_idx_shuffled[split2:]                # ****** 新增 ******
#
# X_train, y_train = X[:split1], y[:split1]
# X_val,   y_val   = X[split1:split2], y[split1:split2]
# X_test,  y_test  = X[split2:], y[split2:]

# ==========================================================
# 方法 2：手动指定训练集编号 -------------------------------
# 若要使用方法 2，把方法 1 这一块注释掉，把下方解除注释即可
# ==========================================================
train_indices =  [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    109, 110, 111, 112
]
all_indices = np.arange(len(X))
remaining_indices = np.setdiff1d(all_indices, train_indices)

# 非训练集编号 → 用于验证+测试
remaining_indices = np.setdiff1d(all_indices, train_indices)

# ==== 新增：打乱剩余样本索引 ====
np.random.seed(42)  # 固定种子以便复现
shuffled_remain = np.random.permutation(remaining_indices)

# 随机分一半做验证集，一半做测试集
val_size = len(shuffled_remain) // 2
val_indices = shuffled_remain[:val_size]
test_indices = shuffled_remain[val_size:]


train_idx, val_idx, test_idx = train_indices, val_indices, test_indices  # ****** 新增 ******

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# ----------- 转换形状 (样本, 时间步, 特征) -----------
X_train = X_train.reshape(-1, 14, 1)
X_val   = X_val.reshape(-1, 14, 1)
X_test  = X_test.reshape(-1, 14, 1)

# ----------- 第4步：构建模型 -----------
model = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(14, 1)),
    MaxPooling1D(2),
    Conv1D(32, 2, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# ----------- 第5步：训练模型 -----------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# ----------- 第6步：训练曲线 -----------
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.pause(0.1)

# ----------- 第7步：评估性能 -----------
y_test_pred  = model.predict(X_test).squeeze()
y_train_pred = model.predict(X_train).squeeze()

train_mae = mean_absolute_error(y_train.squeeze(), y_train_pred)
test_mae  = mean_absolute_error(y_test.squeeze(),  y_test_pred)
abs_err   = np.abs(y_test.squeeze() - y_test_pred)
rel_err   = abs_err / (y_test.squeeze() + 1e-6) * 100

print(f'''
=== 性能报告 ===
训练样本数：{len(y_train)}，测试样本数：{len(y_test)}
训练集 MAE: {train_mae:.4f}
测试集 MAE: {test_mae:.4f}
误差增加: {(test_mae - train_mae) / train_mae * 100:.1f}%

--- 测试集详细误差分析 ---
绝对误差范围: [{np.min(abs_err):.4f}, {np.max(abs_err):.4f}]
平均绝对误差: {np.mean(abs_err):.4f}
相对误差范围: [{np.min(rel_err):.2f}%, {np.max(rel_err):.2f}%]
平均相对误差: {np.mean(rel_err):.2f}%
''')

# ----------- 第8步：可视化预测结果 -----------
orig_id_test = test_idx            # 原始编号（打乱前）

# === 新增：按 orig_id_test 升序重新排序 =========
sort_order   = np.argsort(orig_id_test)
orig_sorted  = orig_id_test[sort_order]
y_test_sorted      = y_test.squeeze()[sort_order]
y_test_pred_sorted = y_test_pred.squeeze()[sort_order]
# ============================================

plt.figure(figsize=(6, 5))
plt.plot(orig_sorted, y_test_sorted,      'bo-', label='True Value')
plt.plot(orig_sorted, y_test_pred_sorted, 'rx--', label='Predicted Value')
plt.title('Test Set Prediction (Original Sample Indices)')
plt.xlabel('Original Sample #')
plt.ylabel('Mass Loss Rate')
plt.legend(); plt.grid(True)

plt.xticks(ticks=orig_sorted, labels=orig_sorted, rotation=90)
plt.ylim(-0.15, 0.3)
plt.tight_layout()
plt.show()
