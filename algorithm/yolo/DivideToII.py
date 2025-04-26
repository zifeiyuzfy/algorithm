import os
import shutil

# 定义路径映射
path_mapping = {
    'D:/YOLO11Traindata/cocox/train/',
    'D:/YOLO11Traindata/cocox/test/',
    'D:/YOLO11Traindata/cocox/valid/'
}

# 遍历每个路径
for path in path_mapping:
    # 检查路径是否存在
    if os.path.exists(path):
        # 创建 labels 和 images 文件夹
        labels_folder = os.path.join(path, 'labels')
        images_folder = os.path.join(path, 'images')
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        # 遍历路径下的所有文件
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            # 检查是否为文件
            if os.path.isfile(file_path):
                if filename.endswith('.txt'):
                    # 移动 txt 文件到 labels 文件夹
                    shutil.move(file_path, os.path.join(labels_folder, filename))
                elif filename.endswith('.jpg'):
                    # 移动 jpg 文件到 images 文件夹
                    shutil.move(file_path, os.path.join(images_folder, filename))
    else:
        print(f"路径 {path} 不存在，请检查。")