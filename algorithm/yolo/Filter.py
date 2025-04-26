import os

# 定义文件路径
path_mapping = {
    'D:/YOLO11Traindata/cocox/train/labels/',
    # 'D:/YOLO11Traindata/cocox/test/labels/',
    'D:/YOLO11Traindata/cocox/valid/labels/'
}

# 遍历每个路径
for path in path_mapping:
    # 检查路径是否存在
    if os.path.exists(path):
        # 遍历路径下的所有文件
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # # 过滤掉开头不为“0”或者“1”的行
                # new_lines = [line for line in lines if line.strip().startswith(('0', '1'))]

                # 过滤掉开头不为“0”的行
                new_lines = [line for line in lines if line.strip().startswith('0')]

                # 将过滤后的内容写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
    else:
        print(f"路径 {path} 不存在，请检查。")