import os
import torch
import numpy as np

# 从 src 包中导入 CrowdCounter 类，用于人群计数
from src.crowd_count import CrowdCounter
# 从 src 包中导入 network 模块，用于加载网络模型
from src import network
# 从 src 包中导入 ImageDataLoader 类，用于加载图像数据
from src.data_loader import ImageDataLoader
# 从 src 包中导入 utils 模块，包含一些工具函数
from src import utils

# 启用 cuDNN 库，它可以加速深度神经网络的计算
torch.backends.cudnn.enabled = True
# 禁用 cuDNN 的自动调优功能，固定算法以保证结果的可重复性
torch.backends.cudnn.benchmark = False
# 是否可视化结果的标志，这里设置为 False 表示不进行可视化
vis = False
# 是否保存输出结果的标志，这里设置为 True 表示保存输出结果
save_output = True

# 测试图像数据的路径
data_path =  './data/original/shanghaitech/part_A_final/test_data/images/'
# 测试数据对应的真实密度图（地面真值）的路径
gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'
# 预训练模型的路径
model_path = './final_models/mcnn_shtechA_660.h5'
# model_path = './final_models/mcnn_shtechB_110.h5'

# 输出结果的目录
output_dir = './output/'
# 从模型路径中提取模型名称
model_name = os.path.basename(model_path).split('.')[0]
# 存储结果的文本文件路径
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# 构建存储密度图的输出目录
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
# 检查存储密度图的目录是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 创建一个 CrowdCounter 实例，用于进行人群计数
net = CrowdCounter()

# 拼接完整的预训练模型路径
trained_model = os.path.join(model_path)
# 调用 network 模块中的 load_net 函数，将预训练模型的参数加载到网络中
network.load_net(trained_model, net)
# 将网络模型移动到 GPU 上进行计算
net.cuda()
# 将网络设置为评估模式，关闭一些在训练时使用的特殊层（如 Dropout）
net.eval()
# 初始化平均绝对误差（MAE）为 0
mae = 0.0
# 初始化均方误差（MSE）为 0
mse = 0.0

# 加载测试数据，使用 ImageDataLoader 类，设置不打乱数据顺序，对真实密度图进行下采样，并且预加载数据
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

# 遍历数据加载器中的每一个数据块
for blob in data_loader:
    # 从数据块中提取图像数据
    im_data = blob['data']
    # 从数据块中提取真实密度图数据
    gt_data = blob['gt_density']
    # 将图像数据和真实密度图数据输入到网络中，得到预测的密度图
    density_map = net(im_data, gt_data)
    # 将预测的密度图从 GPU 移动到 CPU 上，并转换为 numpy 数组
    density_map = density_map.data.cpu().numpy()
    # 计算真实的人群计数，即真实密度图中所有元素的总和
    gt_count = np.sum(gt_data)
    # 计算预测的人群计数，即预测密度图中所有元素的总和
    et_count = np.sum(density_map)
    # 累加当前样本的绝对误差到 MAE 中
    mae += abs(gt_count - et_count)
    # 累加当前样本的平方误差到 MSE 中
    mse += ((gt_count - et_count) * (gt_count - et_count))
    # 如果设置了可视化标志为 True，则调用 utils 模块中的 display_results 函数进行结果可视化
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    # 如果设置了保存输出标志为 True，则调用 utils 模块中的 save_density_map 函数保存预测的密度图
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

# 计算平均绝对误差（MAE），即所有样本的绝对误差总和除以样本数量
mae = mae / data_loader.get_num_samples()
# 计算均方误差（MSE），即所有样本的平方误差总和除以样本数量，然后取平方根
mse = np.sqrt(mse / data_loader.get_num_samples())
# 打印平均绝对误差和均方误差
print ('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))

# 打开存储结果的文本文件，以写入模式打开
f = open(file_results, 'w')
# 将平均绝对误差和均方误差写入文件
f.write('MAE: %0.2f, MSE: %0.2f' % (mae, mse))
# 关闭文件
f.close()