import os

# 定义标签和图像文件夹路径
label_paths = [
    'D:/YOLO11Traindata/cocox/train/labels/',
    # 'D:/YOLO11Traindata/cocox/test/labels/',
    'D:/YOLO11Traindata/cocox/valid/labels/'
]

# 遍历每个标签文件夹
for label_path in label_paths:
    # 找到对应的图像文件夹路径
    image_path = label_path.replace('labels', 'images')
    # 检查标签文件夹是否存在
    if os.path.exists(label_path):
        # 遍历标签文件夹下的所有文件
        for filename in os.listdir(label_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(label_path, filename)
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    # 构建对应的图像文件名
                    image_filename = os.path.splitext(filename)[0] + '.jpg'
                    image_file_path = os.path.join(image_path, image_filename)
                    # 删除空的 txt 文件
                    try:
                        os.remove(file_path)
                        print(f"已删除空文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 时出错: {e}")
                    # 删除对应的 jpg 图像文件
                    if os.path.exists(image_file_path):
                        try:
                            os.remove(image_file_path)
                            print(f"已删除对应的图像文件: {image_file_path}")
                        except Exception as e:
                            print(f"删除图像文件 {image_file_path} 时出错: {e}")
    else:
        print(f"标签文件夹 {label_path} 不存在。")