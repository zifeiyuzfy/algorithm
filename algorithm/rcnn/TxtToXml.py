import os
import xml.dom.minidom
import glob
from PIL import Image
from math import ceil
import shutil

yolo_file = r'E:\YOLO11Traindata\CocoAndVis\test\labels'  # yolo格式下的存放txt标注文件的文件夹
turn_xml_file = r'E:\YOLO11Traindata\CocoAndVis\test\Annotations'  # 转换后储存xml的文件夹地址
img_file = r'E:\YOLO11Traindata\CocoAndVis\test\images'  # 存放图片的文件夹

labels = ['pedestrian','people']
src_img_dir = img_file
src_txt_dir = yolo_file
src_xml_dir = turn_xml_file  # 转换后储存xml的文件夹地址

# 确保目标文件夹存在
if not os.path.exists(src_xml_dir):
    os.makedirs(src_xml_dir)

img_Lists = glob.glob(src_img_dir + '/*.jpg')
img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))  # os.path.basename返回path最后的文件名

img_names = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)  # os.path.splitext(“文件路径”)    分离文件名与扩展名
    img_names.append(temp1)

total_num = len(img_names)  # 统计当前总共要转换的图片标注数量
count = 0  # 计数变量
for img in img_names:  # 这里的img是不加后缀的图片名称，如：'GF3_SAY_FSI_002732_E122.3_N29.9_20170215_L1A_HH_L10002188179__1__4320___10368'
    count += 1
    if count % 1000 == 0:
        print("当前转换进度{}/{}".format(count, total_num))
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size

    txt_file_path = src_txt_dir + '/' + img + '.txt'
    if os.path.exists(txt_file_path):
        # 打开yolo格式下的txt文件
        gt = open(txt_file_path).read().splitlines()
        if gt:
            # 将主干部分写入xml文件中
            xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>VOC2007</folder>\n')
            xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(width) + '</width>\n')
            xml_file.write('        <height>' + str(height) + '</height>\n')
            xml_file.write('        <depth>3</depth>\n')
            xml_file.write('    </size>\n')

            # write the region of image on xml file
            for img_each_label in gt:
                spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + str(labels[int(spt[0])]) + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')

                center_x = round(float(spt[1].strip()) * width)
                center_y = round(float(spt[2].strip()) * height)
                bbox_width = round(float(spt[3].strip()) * width)
                bbox_height = round(float(spt[4].strip()) * height)
                xmin = str(int(center_x - bbox_width / 2))
                ymin = str(int(center_y - bbox_height / 2))
                xmax = str(int(center_x + bbox_width / 2))
                ymax = str(int(center_y + bbox_height / 2))

                xml_file.write('            <xmin>' + xmin + '</xmin>\n')
                xml_file.write('            <ymin>' + ymin + '</ymin>\n')
                xml_file.write('            <xmax>' + xmax + '</xmax>\n')
                xml_file.write('            <ymax>' + ymax + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')

            xml_file.write('</annotation>')
        else:
            # 将主干部分写入xml文件中
            xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>VOC2007</folder>\n')
            xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(width) + '</width>\n')
            xml_file.write('        <height>' + str(height) + '</height>\n')
            xml_file.write('        <depth>3</depth>\n')
            xml_file.write('    </size>\n')
            xml_file.write('</annotation>')
    else:
        print(f"未找到对应的标注文件: {txt_file_path}，跳过该图像的处理。")
