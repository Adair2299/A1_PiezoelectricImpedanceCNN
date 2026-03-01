# 整合写入ban-1，并创建"sasq数据整合.xlsx"

import os
import openpyxl
import csv

# 文件夹路径（CSV文件所在目录）
folder_path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\04.23实验数据汇总\实验数据汇总\形貌参数数据\形貌参数Sa,Sq数据\ban-1"

# 输出Excel文件的保存路径
output_dir = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\04.23实验数据汇总\实验数据汇总\形貌参数数据\形貌参数Sa,Sq数据"
output_file = os.path.join(output_dir, "sasq数据整合.xlsx")

# 创建新的Excel工作簿
output_wb = openpyxl.Workbook()
output_ws = output_wb.active
output_ws.title = "SaSq数据"

# 设置表头
output_ws['A1'] = "文件编号"
output_ws['A2'] = "Sq(B16)"
output_ws['A3'] = "Sa(B17)"

# 从01到30编号
for i in range(1, 31):
    # 构建文件名，确保是两位数格式
    filename = f"{i:02d}.csv"
    file_path = os.path.join(folder_path, filename)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {filename} 不存在，跳过")
        continue

    # 读取CSV文件
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        data = list(csv_reader)

        # 提取B16和B17的数据（CSV是0-based索引，B16是第15行第1列）
        try:
            b16_value = data[15][1] if len(data) > 15 and len(data[15]) > 1 else ""
            b17_value = data[16][1] if len(data) > 16 and len(data[16]) > 1 else ""
        except IndexError:
            print(f"文件 {filename} 格式不符合预期")
            b16_value = ""
            b17_value = ""

        # 写入到Excel中
        col_letter = openpyxl.utils.get_column_letter(i + 1)  # B列是2，所以i+1
        output_ws[f'{col_letter}1'] = f"{i:02d}"  # 文件编号
        output_ws[f'{col_letter}2'] = b16_value  # B16数据
        output_ws[f'{col_letter}3'] = b17_value  # B17数据

# 保存Excel文件
"""
必须保持sasq数据整合.xlsx没被其他应用打开！
"""
output_wb.save(output_file)
print(f"数据已保存到 {output_file}")