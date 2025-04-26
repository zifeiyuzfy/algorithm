import os
import torch
import numpy as np
import cv2

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

# 输入图片文件夹路径
input_path = './img/'
# 输出图片文件夹路径
output_path = './imgout/'
# 确保输出文件夹存在
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 预训练模型的路径
model_path = './final_models/mcnn_shtechB_110.h5'

net = CrowdCounter()
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()

image_filename = 'people3.jpg'  # 实际要处理的图片文件名
img_path = os.path.join(input_path, image_filename)

# 读取图片
im = cv2.imread(img_path)
if im is None:
    print(f"无法读取图片: {img_path}")
else:
    # 将彩色图像转换为灰度图像
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 增加一个维度，使其变为 [1, height, width] 的形状
    im = np.expand_dims(im, axis=0)
    # 再增加一个通道维度，变为 [1, 1, height, width]
    im = np.expand_dims(im, axis=1)
    # 转换为模型输入所需的格式
    im = torch.from_numpy(im).float().cuda()

    # 进行人群计数
    density_map = net(im)
    density_map = density_map.data.cpu().numpy()
    et_count = np.sum(density_map)

    # 对密度图进行归一化处理，使其范围在 0-255 之间，方便保存为图像
    density_map = density_map.squeeze()  # 移除多余的维度
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min()) * 255
    density_map = density_map.astype(np.uint8)

    # 应用伪彩色映射
    colored_density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)

    # 添加文字标注
    text = f'Total people: {et_count:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # 调小字体大小
    color = (0, 255, 0)  # 绿色
    thickness = 1  # 调小字体粗细
    position = (10, 20)  # 左上角位置，适当调整
    cv2.putText(colored_density_map, text, position, font, font_scale, color, thickness)

    # 生成处理后的图片名
    output_filename = f'test_{image_filename}'
    output_img_path = os.path.join(output_path, output_filename)
    # 保存伪彩色密度图
    cv2.imwrite(output_img_path, colored_density_map)

    print(f"处理图片 {image_filename} 完成，预测人数: {et_count:.2f}")
