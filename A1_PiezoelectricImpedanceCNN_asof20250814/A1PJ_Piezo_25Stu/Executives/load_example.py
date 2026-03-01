
# ============== 数据加载示例 ==============
import numpy as np

# 加载质量损失数据 (形状: [n_samples, 1])
mass_loss = np.load('mass_loss_rate.npy') 

# 加载阻抗谱数据 (形状: [n_samples, n_features])
impedance = np.load('Imp1_500.npy')  

print(f"质量损失数据形状: {mass_loss.shape}")
print(f"阻抗谱数据形状: {impedance.shape}")
