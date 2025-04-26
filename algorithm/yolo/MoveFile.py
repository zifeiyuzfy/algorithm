import os
import shutil

# 定义源路径和目标路径的映射
path_mapping = {
    # 'D:/YOLO11Traindata/VOCdevkit/train/images/': 'D:/YOLO11Traindata/HatAndFire/train/images/',
    'D:/YOLO11Traindata/VOCdevkit/train/labels/': 'D:/YOLO11Traindata/HatAndFire/train/labels/',
    'D:/YOLO11Traindata/VOCdevkit/test/images/': 'D:/YOLO11Traindata/HatAndFire/test/images/',
    'D:/YOLO11Traindata/VOCdevkit/test/labels/': 'D:/YOLO11Traindata/HatAndFire/test/labels/',
    'D:/YOLO11Traindata/VOCdevkit/val/images/': 'D:/YOLO11Traindata/HatAndFire/valid/images/',
    'D:/YOLO11Traindata/VOCdevkit/val/labels/': 'D:/YOLO11Traindata/HatAndFire/valid/labels/'
}

# 遍历路径映射
for src_path, dst_path in path_mapping.items():
    # 确保目标路径存在，如果不存在则创建
    # if not os.path.exists(dst_path):
    #     os.makedirs(dst_path)
    #     print(f"Created directory: {dst_path}")

    # 遍历源路径下的所有文件
    for filename in os.listdir(src_path):
        # 构造源文件和目标文件的完整路径
        src_file = os.path.join(src_path, filename)
        dst_file = os.path.join(dst_path, filename)

        # 迁移文件
        shutil.move(src_file, dst_file)
        print(f"Moved: {src_file} -> {dst_file}")