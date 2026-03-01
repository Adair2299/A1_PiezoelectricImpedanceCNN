# 继续写入ban-2到pzt-5，追加写入"sasq数据整合.xlsx"

import os
import openpyxl
import csv

for n in range(1, 6):

    num=int(n+5)

    # 文件夹路径（CSV文件所在目录）
    folder_path = rf"E:\01我的\大三下(202501-202508)\大创-压电阻抗\04.23实验数据汇总\实验数据汇总\形貌参数数据\形貌参数Sa,Sq数据\pzt-{n}"

    # 输出Excel文件的路径（必须是已存在的文件）
    output_dir = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\04.23实验数据汇总\实验数据汇总\形貌参数数据\形貌参数Sa,Sq数据"
    output_file = os.path.join(output_dir, "sasq数据整合.xlsx")

    # 检查文件是否存在，如果不存在则报错
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"目标文件 {output_file} 不存在，请先创建或检查路径！")

    # 打开现有的Excel文件（不新建）
    output_wb = openpyxl.load_workbook(output_file)
    output_ws = output_wb.active  # 默认操作第一个Sheet

    # 确保Sheet名称正确（可选）
    if output_ws.title != "SaSq数据":
        print(f"警告：当前Sheet名称是 '{output_ws.title}'，可能不是预期的工作表。")

    # 设置表头
    output_ws[f'A{3*num-2}'] = f"pzt-{n}"
    output_ws[f'A{3*num-1}'] = "Sq(B16)"
    output_ws[f'A{3*num-0}'] = "Sa(B17)"

    # 从01.csv到30.csv提取数据并写入B5:AE6
    for i in range(1, 31):
        filename = f"{i:02d}.csv"
        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            print(f"文件 {filename} 不存在，跳过")
            continue

        with open(file_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = list(csv_reader)

            try:
                b16_value = data[15][1] if len(data) > 15 and len(data[15]) > 1 else ""
                b17_value = data[16][1] if len(data) > 16 and len(data[16]) > 1 else ""
            except IndexError:
                print(f"文件 {filename} 格式不符合预期")
                b16_value = ""
                b17_value = ""

            col_num = i + 1  # B列=2, C列=3, ..., AE列=31
            col_letter = openpyxl.utils.get_column_letter(col_num)

            # 写入B5:AE5（B16数据）
            output_ws[f'{col_letter}{3*num-1}'] = b16_value
            # 写入B6:AE6（B17数据）
            output_ws[f'{col_letter}{3*num}'] = b17_value

    # 保存修改（不会覆盖原有数据，只会更新B5:AE6）
    """
    必须保持sasq数据整合.xlsx没被其他应用打开！
    """
    output_wb.save(output_file)
    print(f"数据已追加到 {output_file} 的 B{3*num-1}:AE{3*num}，原有数据保持不变。")