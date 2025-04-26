import cv2  # OpenCV库，用于图像处理和视频操作
import os  # 导入os模块，用于路径操作
from ultralytics import YOLO  # YOLO模型库，用于目标检测
import numpy as np  # 用于数学运算

# 定义预测函数，用于对图像进行目标检测
def predict(chosen_model, img, classes=[], conf=0.5, iou=0.5):
    """
    使用YOLO模型对图像进行目标检测。
    参数:
        chosen_model: YOLO模型实例。
        img: 输入图像。
        classes: 指定检测的类别列表（可选，默认为空列表）。
        conf: 检测的置信度阈值（默认为0.7）。
        iou: NMS的IoU阈值（默认为0.5）。

    返回:
        results: 检测结果。
    """
    if classes:  # 如果指定了类别
        results = chosen_model.predict(img, classes=classes, conf=conf, iou=iou)  # 按指定类别和置信度检测
    else:  # 如果未指定类别
        results = chosen_model.predict(img, conf=conf, iou=iou)  # 检测所有类别
    return results

# 定义平滑检测框位置的函数
def smooth_box(box, alpha=0.5):
    """
    使用指数加权平均平滑检测框位置。
    参数:
        box: 当前检测框。
        alpha: 平滑系数（0 < alpha < 1）。
    返回:
        平滑后的检测框坐标。
    """
    box_id = int(box.cls[0])  # 使用类别ID作为键
    if box_id not in smoothed_boxes:
        smoothed_boxes[box_id] = box.xyxy[0]  # 初始化为当前框的位置
    else:
        # 使用指数加权平均来平滑位置
        smoothed_boxes[box_id] = alpha * smoothed_boxes[box_id] + (1 - alpha) * box.xyxy[0]
    return smoothed_boxes[box_id]

# 定义预测并绘制检测框的函数
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, iou=0.5, rectangle_thickness=4, text_thickness=2, font_scale=1.5):
    """
    对图像进行目标检测，并在检测到的目标周围绘制矩形框和类别名称及置信度。

    参数:
        chosen_model: YOLO模型实例。
        img: 输入图像。
        classes: 指定检测的类别列表（可选，默认为空列表）。
        conf: 检测的置信度阈值（默认为0.7）。
        iou: NMS的IoU阈值（默认为0.5）。
        rectangle_thickness: 矩形框的厚度（默认为4）。
        text_thickness: 文本的厚度（默认为2）。
        font_scale: 字体大小（默认为1.5）。

    返回:
        img: 绘制了检测框和类别名称及置信度的图像。
        results: 检测结果。
    """
    results = predict(chosen_model, img, classes, conf=conf, iou=iou)  # 调用预测函数获取检测结果
    for result in results:  # 遍历检测结果
        for box in result.boxes:  # 遍历每个检测框
            # 平滑检测框位置
            smoothed_xyxy = smooth_box(box)
            # 绘制矩形框
            cv2.rectangle(img,
                          (int(smoothed_xyxy[0]), int(smoothed_xyxy[1])),  # 左上角坐标
                          (int(smoothed_xyxy[2]), int(smoothed_xyxy[3])),  # 右下角坐标
                          (0, 255, 0),  # 矩形框颜色（绿色）
                          rectangle_thickness)  # 矩形框厚度

            # 在矩形框上方绘制类别名称和置信度
            class_name = result.names[int(box.cls[0])]  # 获取类别名称
            confidence = float(box.conf[0])  # 获取置信度
            label = f"{class_name} {confidence:.2f}"  # 格式化标签文本
            cv2.putText(img,
                        label,  # 显示类别名称和置信度
                        (int(smoothed_xyxy[0]), int(smoothed_xyxy[1]) - 10),  # 文本位置
                        cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型（更清晰的字体）
                        font_scale,  # 字体大小
                        (0, 255, 0),  # 文本颜色（绿色）
                        text_thickness)  # 文本厚度
    return img, results

# 定义创建视频写入器的函数
def create_video_writer(video_cap, output_filename):
    """
    创建一个用于写入视频的VideoWriter对象。

    参数:
        video_cap: 视频捕获对象（cv2.VideoCapture）。
        output_filename: 输出视频文件名。

    返回:
        writer: VideoWriter对象，用于将处理后的帧写入视频文件。
    """
    # 获取视频流的宽度、高度和帧率
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # 初始化FourCC和视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义视频编码格式
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

# 初始化平滑检测框的字典
smoothed_boxes = {}

# 加载YOLO模型
model = YOLO('runs/detect/train/weights/best.pt')  # 加载训练好的YOLO模型权重文件

# 定义输入视频路径
video_path = r"video/people.mp4"

# 解析输入视频路径
video_dir = os.path.dirname(video_path)  # 获取输入视频所在的目录
video_name = os.path.basename(video_path)  # 获取输入视频的文件名（带扩展名）

# 修改文件名，在原文件名前加上 "test"
output_filename = f"test_{video_name}"

# 打开输入视频文件
cap = cv2.VideoCapture(video_path)

# 创建视频写入器，用于将处理后的视频帧写入输出文件
writer = create_video_writer(cap, os.path.join(video_dir, output_filename))

# 循环读取视频帧
while True:
    success, img = cap.read()  # 读取一帧
    if not success:  # 如果读取失败，退出循环
        break
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5, iou=0.5)
    writer.write(result_img)  # 写入处理后的帧
    cv2.imshow("Image", result_img)  # 显示处理后的帧
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

# 释放资源
writer.release()
cap.release()
cv2.destroyAllWindows()