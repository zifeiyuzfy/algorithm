import os

# 定义六个路径
paths = [
    'D:/YOLO11Traindata/FireYOLO11/train/images/',
    # 'D:/YOLO11Traindata/FireYOLO11/train/labels/',
    'D:/YOLO11Traindata/FireYOLO11/test/images/',
    # 'D:/YOLO11Traindata/FireYOLO11/test/labels/',
    'D:/YOLO11Traindata/FireYOLO11/valid/images/',
    # 'D:/YOLO11Traindata/FireYOLO11/valid/labels/'
]

# 遍历每个路径
for path in paths:
    # 遍历路径下的所有文件
    for filename in os.listdir(path):
        # 检查文件是否以 .txt 或 .jpg 结尾
        if (filename.endswith(".txt") or filename.endswith(".jpg") or filename.endswith(".JPG")) and not filename.startswith("fire"):
            # 构造新的文件名
            new_filename = "fire" + filename
            # 获取文件的完整路径
            old_file = os.path.join(path, filename)
            new_file = os.path.join(path, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed: {os.path.join(path, filename)} -> {os.path.join(path, new_filename)}")