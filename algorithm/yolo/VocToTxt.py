# 作者：CSDN-笑脸惹桃花 https://blog.csdn.net/qq_67105081?type=blog
# github:peng-xiaobai https://github.com/peng-xiaobai/Dataset-Conversion

import os
import xml.etree.ElementTree as ET

# 定义类别顺序
categories = ['hat', 'nohat']
category_to_index = {category: index for index, category in enumerate(categories)}

# 定义输入文件夹和输出文件夹
input_folder = r'D:\hat\hatdata\Annotations'  # 替换为实际的XML文件夹路径
output_folder = r'D:\hat\hatdata\labels'  # 替换为实际的输出TXT文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有XML文件
for filename in os.listdir(input_folder):
    if filename.endswith('.xml'):
        xml_path = os.path.join(input_folder, filename)
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 提取图像的尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        # 存储name和对应的归一化坐标
        objects = []

        # 遍历XML中的object标签
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in category_to_index:
                category_index = category_to_index[name]
            else:
                continue  # 如果name不在指定类别中，跳过该object

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # 转换为中心点坐标和宽高
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin

            # 归一化
            x = x_center / width
            y = y_center / height
            w = w / width
            h = h / height

            objects.append(f"{category_index} {x} {y} {w} {h}")

        # 输出结果到对应的TXT文件
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, 'w') as f:
            for obj in objects:
                f.write(obj + '\n')