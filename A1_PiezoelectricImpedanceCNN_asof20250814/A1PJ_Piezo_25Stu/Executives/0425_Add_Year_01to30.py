# 在实验数据excel文件名前加年份，01到30

import os
import re

# 设置基础目录路径
base_directory = r"E:\01我的\大三下(202501-202508)\大创-压电阻抗\实验数据汇总\阻抗数据--总\1-500 xlsx"

# 处理从01到30的子文件夹
for folder_num in range(1, 31):
    # 格式化为两位数，如01, 02,...30
    folder_name = f"{folder_num:02d}"
    directory = os.path.join(base_directory, folder_name)

    # 检查子文件夹是否存在
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        continue

    print(f"正在处理目录: {directory}")

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        # 匹配文件名模式
        match = re.match(r"1-500 (\d{2})\.(\d{2})_(\d{2})\.xlsx", filename)
        if match:
            month = int(match.group(1))
            day = match.group(2)
            file_num = match.group(3)  # 获取文件末尾的数字

            # 确定年份部分
            if month == 12:
                new_prefix = "24"
            else:  # 1-4月
                new_prefix = "25"

            # 构建新文件名
            new_filename = f"1-500 {new_prefix}.{match.group(1)}.{match.group(2)}_{file_num}.xlsx"

            # 重命名文件
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"重命名失败 {filename}: {e}")

print("所有文件重命名完成！")