# 作者：CSDN-笑脸惹桃花 https://blog.csdn.net/qq_67105081?type=blog
# github:peng-xiaobai https://github.com/peng-xiaobai/Dataset-Conversion
import os
import shutil
import random


# random.seed(0)  #随机种子，可自选开启
def split_data(file_path, label_path, new_file_path, train_rate, val_rate, test_rate):
    images = os.listdir(file_path)
    labels = os.listdir(label_path)
    images_no_ext = {os.path.splitext(image)[0]: image for image in images}
    labels_no_ext = {os.path.splitext(label)[0]: label for label in labels}
    matched_data = [(img, images_no_ext[img], labels_no_ext[img]) for img in images_no_ext if img in labels_no_ext]

    unmatched_images = [img for img in images_no_ext if img not in labels_no_ext]
    unmatched_labels = [label for label in labels_no_ext if label not in images_no_ext]
    if unmatched_images:
        print("未匹配的图片文件:")
        for img in unmatched_images:
            print(images_no_ext[img])
    if unmatched_labels:
        print("未匹配的标签文件:")
        for label in unmatched_labels:
            print(labels_no_ext[label])

    random.shuffle(matched_data)
    total = len(matched_data)
    train_data = matched_data[:int(train_rate * total)]
    val_data = matched_data[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_data = matched_data[int((train_rate + val_rate) * total):]

    # 处理训练集
    for img_name, img_file, label_file in train_data:
        old_img_path = os.path.join(file_path, img_file)
        old_label_path = os.path.join(label_path, label_file)
        new_img_dir = os.path.join(new_file_path, 'train', 'images')
        new_label_dir = os.path.join(new_file_path, 'train', 'labels')
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)
        shutil.copy(old_img_path, os.path.join(new_img_dir, img_file))
        shutil.copy(old_label_path, os.path.join(new_label_dir, label_file))
    # 处理验证集
    for img_name, img_file, label_file in val_data:
        old_img_path = os.path.join(file_path, img_file)
        old_label_path = os.path.join(label_path, label_file)
        new_img_dir = os.path.join(new_file_path, 'val', 'images')
        new_label_dir = os.path.join(new_file_path, 'val', 'labels')
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)
        shutil.copy(old_img_path, os.path.join(new_img_dir, img_file))
        shutil.copy(old_label_path, os.path.join(new_label_dir, label_file))
    # 处理测试集
    for img_name, img_file, label_file in test_data:
        old_img_path = os.path.join(file_path, img_file)
        old_label_path = os.path.join(label_path, label_file)
        new_img_dir = os.path.join(new_file_path, 'test', 'images')
        new_label_dir = os.path.join(new_file_path, 'test', 'labels')
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)
        shutil.copy(old_img_path, os.path.join(new_img_dir, img_file))
        shutil.copy(old_label_path, os.path.join(new_label_dir, label_file))
    print("数据集已划分完成")


if __name__ == '__main__':
    file_path = r"D:\hat\hatdata\JPEGImages"  # 图片文件夹
    label_path = r'D:\hat\hatdata\labels'  # 标签文件夹
    new_file_path = r"D:\hat\VOCdevkit"  # 新数据存放位置
    split_data(file_path, label_path, new_file_path, train_rate=0.8, val_rate=0.1, test_rate=0.1)