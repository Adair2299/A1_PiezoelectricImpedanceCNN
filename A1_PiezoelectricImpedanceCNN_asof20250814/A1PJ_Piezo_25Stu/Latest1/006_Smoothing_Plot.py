from FunA_Adapt_Smoothing import adaptive_smooth
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Qt5Agg')

# 加载数据
Imp1_500 = np.load('../Latest1/Imp1_500.npy')  # (2496,116)
mass_loss_rate = np.load('../Latest1/mass_loss_rate.npy')

# 自适应平滑
Imp1_500_smooth = adaptive_smooth(Imp1_500, base_sigma=20, window=50, alpha=25)

# 设置字体
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10

# 创建频率轴 (1-500 kHz, 步长0.2)
frequency = np.arange(1, 500.2, 0.2)  # 生成1, 1.2, 1.4,..., 500.0

# 确保数据长度与频率点数匹配
assert len(frequency) == Imp1_500.shape[0], "频率点数与数据长度不匹配"

# 绘图
plt.figure(figsize=(5, 4))  # 设置图形大小
plt.scatter(frequency, Imp1_500[:, 82], label='Origin', s=0.35, alpha=0.5, c='blue', edgecolors='none')  # 去除点边缘线 # alpha=0.05
plt.plot(frequency, Imp1_500_smooth[:, 82], label='Adaptive Smoothed', c='red', linewidth=0.5, alpha=0.6)

# 设置x轴标签和范围
plt.xlabel('Frequency (kHz)')
plt.ylabel('Conductance (mS)')
plt.xlim(1, 500)  # 确保x轴范围是1-500

# 优化x轴刻度显示
plt.xticks(np.arange(0, 501, 50))  # 每50kHz显示一个主要刻度

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
plt.tight_layout()  # 自动调整布局


# 保存为600dpi的JPG文件
output_path = r'E:\01我的\大三下(202501-202508)\大创-压电阻抗\论文材料\绘图\P26_M45_平滑演示\P26_M45_Sigma20_Window50_Alpha25.jpg'
plt.savefig(output_path, dpi=600, format='jpg')
plt.show(block=True)
