# 将各个钢板的数据混到一个表里
# IntegrateExcel和IntegrateExcel2两个要先后全部使用

import pandas as pd
import os

# 设置文件夹路径
folder_path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\14天阻抗汇总\Term2"   # 要改！
output_file = os.path.join(folder_path, "Term4 1-500 01.19.xlsx")   # 要改！

# 存储所有列的内容（每列是1501个数据）
columns_data = []

for i in range(1, 31):
    # 如果 i == 6，直接跳出循环（直接忽略6号钢板）
    if i == 6:
        continue  # 终止循环，不再处理后续文件

    file_name = f"1-500 01.19_{i:02d}.xlsx"  # 要改！
    file_path = os.path.join(folder_path, file_name)

    # 读取整列B
    df = pd.read_excel(file_path, usecols="B")

    # 取B2:B1502（对应iloc[1:2496]）
    col_data = df.iloc[1:2497].reset_index(drop=True)
    columns_data.append(col_data)

# 将每列合并为一个新DataFrame
merged_df = pd.concat(columns_data, axis=1)

# 在最上面插入一行空值（1501行变1502行，从第2行开始放数据）
merged_df.loc[-1] = [None] * merged_df.shape[1]  # 添加一行空
merged_df.index = merged_df.index + 1           # 所有行向下移动
merged_df = merged_df.sort_index()              # 重排序

# 保存新文件，不保留列名
merged_df.to_excel(output_file, index=False, header=False)

print("✅ 合并完成！已保存至：", output_file)
