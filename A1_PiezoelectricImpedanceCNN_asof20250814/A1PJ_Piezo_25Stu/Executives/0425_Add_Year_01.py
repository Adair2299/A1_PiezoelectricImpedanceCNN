# 在实验数据excel文件名前加年份,仅对01

import os
import re

# 设置目录路径
directory = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\实验数据汇总\阻抗数据--总\1-500 xlsx\01"

# 遍历目录中的文件
for filename in os.listdir(directory):
    # 匹配文件名模式
    match = re.match(r"1-500 (\d{2})\.(\d{2})_01\.xlsx", filename)
    if match:
        month = int(match.group(1))
        day = match.group(2)

        # 确定年份部分
        if month == 12:
            new_prefix = "24"
        else:  # 1-4月
            new_prefix = "25"

        # 构建新文件名
        new_filename = f"1-500 {new_prefix}.{match.group(1)}.{match.group(2)}_01.xlsx"

        # 重命名文件
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print("文件重命名完成！")