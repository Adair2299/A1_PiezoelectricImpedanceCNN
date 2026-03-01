import pandas as pd
from pathlib import Path

# 1. 定义路径模板
input_template = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\有限元\PZT FEM2\{pnum:02d}\整合数据_按列排列{pnum:02d}.xlsx"
output_path = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\有限元\PZT FEM2\整合数据_全部合并_顺序标题.xlsx"

# 2. 初始化合并数据和标题列表
merged_data = pd.DataFrame()
all_headers = []  # 存储所有列的标题

# 3. 遍历所有pnum值（1-17, 27, 28, 30）
pnum_list = list(range(1, 18)) + [27, 28, 30]

for pnum in pnum_list:
    file_path = input_template.format(pnum=pnum)

    if not Path(file_path).exists():
        print(f"警告：文件 {file_path} 不存在，已跳过")
        continue

    try:
        # 读取A2:E2497数据（无标题）
        df = pd.read_excel(file_path, header=None, usecols="A:E", skiprows=1, nrows=2496)

        # 生成当前文件的5列标题 (P{pnum:02}T0 到 P{pnum:02}T4)
        headers = [f'P{pnum:02}T{j}' for j in range(5)]
        all_headers.extend(headers)  # 添加到总标题列表

        # 将数据添加到合并DataFrame
        for col in range(5):
            merged_data[f'P{pnum:02}T{col}'] = df[col]

    except Exception as e:
        print(f"处理文件 {file_path} 失败：{e}")

# 4. 保存到Excel（标题在第1行，数据从第2行开始）
try:
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 写入标题行
        pd.DataFrame([all_headers]).to_excel(writer, index=False, header=False, startrow=0)
        # 写入数据
        merged_data.to_excel(writer, index=False, header=False, startrow=1)

    print(f"合并完成！结果保存至：{output_path}")
    print(f"总列数：{len(all_headers)}（{len(pnum_list)}个文件 × 5列）")
except Exception as e:
    print(f"保存失败：{e}")