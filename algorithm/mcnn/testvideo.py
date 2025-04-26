import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network, utils
from src.data_loader import ImageDataLoader

# 初始化全局模型（避免重复加载）
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
net = CrowdCounter()
trained_model = './final_models/mcnn_shtechA_660.h5'  # 可改为动态传入
network.load_net(trained_model, net)
net.cuda()
net.eval()


def estimate_density_map(video_frame):
    """
    根据输入视频帧返回密度图及估计人数

    参数:
        video_frame: numpy数组，形状为(H, W, 3)，RGB格式，像素值范围[0, 255]

    返回:
        density_map: 密度图 numpy数组 (1, 1, H', W')
        person_count: 估计人数（密度图积分结果）
    """
    # 预处理视频帧
    frame = utils.preprocess_image(video_frame)  # 假设存在预处理函数，需根据模型输入要求实现
    # 转换为模型输入格式 (N, C, H, W)
    input_tensor = torch.from_numpy(frame).unsqueeze(0).cuda()

    # 前向传播
    with torch.no_grad():
        density_map = net(input_tensor)

    # 转换为numpy并计算人数
    density_map = density_map.data.cpu().numpy()
    person_count = np.sum(density_map)

    return density_map, person_count


# 原主流程（添加函数调用示例）
if __name__ == "__main__":
    vis = False
    save_output = True
    data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
    gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'
    output_dir = './output/density_maps_mcnn_shtechA'

    # 初始化数据加载器（用于测试集评估）
    data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    mae = 0.0
    mse = 0.0

    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        video_frame = im_data[0].permute(1, 2, 0).cpu().numpy()  # 示例：将张量转换为视频帧格式

        # 调用新增函数
        density_map, et_count = estimate_density_map(video_frame)

        gt_count = np.sum(gt_data)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) ** 2)

        # 原有保存/可视化逻辑...

    mae /= data_loader.get_num_samples()
    mse = np.sqrt(mse / data_loader.get_num_samples())
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}")


# 补充：建议添加的预处理函数（需根据模型实际输入调整）
def preprocess_image(frame):
    """
    视频帧预处理函数（示例，需根据MCNN输入要求调整）
    假设模型输入为RGB图像，归一化至[0,1]，尺寸适配模型要求
    """
    # 调整尺寸（假设模型输入尺寸为原图，若需要resize请添加）
    # frame = cv2.resize(frame, (model_width, model_height))

    # 转换颜色通道 (H,W,C) -> (C,H,W)
    frame = frame.transpose((2, 0, 1))

    # 归一化（假设模型输入范围0-1）
    frame = frame.astype(np.float32) / 255.0

    # 添加批次维度（N,C,H,W）
    return frame