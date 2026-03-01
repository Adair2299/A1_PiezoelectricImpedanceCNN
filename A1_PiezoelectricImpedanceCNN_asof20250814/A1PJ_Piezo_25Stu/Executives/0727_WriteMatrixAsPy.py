"""
把质量损失和阻抗写成py文件
"""
import numpy as np
import pandas as pd
from pathlib import Path


# ------------ 保存质量损失数据 ------------
def save_mass_loss():
    folder = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\数据分析\数据操作")
    file = "3_sasq和质量损失率.xlsx"
    file_path = folder / file

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取四个周期的数据
    df1 = pd.read_excel(file_path, header=None, skiprows=15, nrows=1, usecols=range(1, 31))
    df2 = pd.read_excel(file_path, header=None, skiprows=23, nrows=1, usecols=range(1, 31))
    df3 = pd.read_excel(file_path, header=None, skiprows=31, nrows=1, usecols=range(1, 31))
    df4 = pd.read_excel(file_path, header=None, skiprows=39, nrows=1, usecols=range(1, 31))

    # 合并并处理数据
    result = pd.concat([df1, df2, df3, df4], ignore_index=True)
    result = np.array(result)
    result = np.delete(result, [3,5,17], axis=1)  # 删除6号钢板数据
    mass_loss_rate = result.reshape(-1, 1)

    # 保存为.npy文件
    np.save('../Latest2/mass_loss_rate_RemoveWrongEMI.npy', mass_loss_rate)
    print(f"质量损失数据已保存为 mass_loss_rate_RemoveWrongEMI.npy，形状: {mass_loss_rate.shape}")


# ------------ 保存阻抗谱数据 ------------
def save_impedance():
    file_path = Path(r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term all 1-500 RemoveWrongEMI.xlsx")

    try:
        df = pd.read_excel(file_path, usecols="B:DE", header=None, skiprows=1, nrows=2496)
        Imp1_500 = df.values
        np.save('../Latest2/Imp1_500_RemoveWrongEMI.npy', Imp1_500)
        print(f"阻抗谱数据已保存为 Imp1_500_RemoveWrongEMI.npy，形状: {Imp1_500.shape}")
    except Exception as e:
        print(f"读取失败: {e}")


# ------------ 执行保存 ------------
if __name__ == "__main__":
    save_mass_loss()
    save_impedance()

    # 生成使用示例代码
    example_code = '''
# ============== 数据加载示例 ==============
import numpy as np

# 加载质量损失数据 (形状: [n_samples, 1])
mass_loss = np.load('mass_loss_rate.npy') 

# 加载阻抗谱数据 (形状: [n_samples, n_features])
impedance = np.load('Imp1_500.npy')  

print(f"质量损失数据形状: {mass_loss.shape}")
print(f"阻抗谱数据形状: {impedance.shape}")
'''
    with open('load_example.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    print("已生成数据加载示例脚本: load_example.py")