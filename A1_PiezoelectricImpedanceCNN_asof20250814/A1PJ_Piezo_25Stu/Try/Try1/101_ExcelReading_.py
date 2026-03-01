import pandas as pd
import numpy as np
import os

# 定义目标文件夹路径
folder_path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term1"

# 创建空矩阵容器
Imp1_500 = []

# 遍历30个文件
for i in range(1, 31):
    # 生成带两位序号的文件名（01-30）
    filename = f"1-500 12.24_{i:02d}.xlsx"
    file_path = os.path.join(folder_path, filename)

    # 读取Excel的B列数据（B2-B2497对应索引1-2496）
    df = pd.read_excel(file_path, usecols="B", header=None, skiprows=1, nrows=2496)

    # 转换为向量并添加到矩阵
    Imp1_500.append(df.iloc[:, 0].values)

# 转换为numpy矩阵（30行×2496列）
Imp1_500 = np.array(Imp1_500)

# 验证结果
print(Imp1_500)
print(f"矩阵维度：{Imp1_500.shape}")  # 应输出 (30, 2496)