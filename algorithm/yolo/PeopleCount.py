import cv2
import os
from ultralytics import YOLO
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# 初始化 DeepSort 跟踪器
tracker = DeepSort(max_age=10)  # 减少 max_age 可使目标丢失后更快移除

# 定义平滑检测框位置的函数
smoothed_boxes = {}  # 用于存储平滑后的检测框位置

def smooth_box(track, alpha=0.7):
    """
    使用指数加权平均平滑检测框位置。
    参数:
        track: DeepSort 的 Track 对象。
        alpha: 平滑系数（0 < alpha < 1）。
    返回:
        平滑后的检测框坐标 [x1, y1, x2, y2]。
    """
    track_id = track.track_id  # 使用 track_id 作为键
    bbox = track.to_ltrb()  # 获取当前检测框坐标 [x1, y1, x2, y2]

    if track_id not in smoothed_boxes:
        smoothed_boxes[track_id] = np.array(bbox)  # 初始化为当前框的位置
    else:
        # 使用指数加权平均来平滑位置
        smoothed_boxes[track_id] = alpha * smoothed_boxes[track_id] + (1 - alpha) * np.array(bbox)
    return smoothed_boxes[track_id].astype(int)  # 返回整数坐标

# 定义预测函数
def predict(chosen_model, img, classes=[], conf=0.5, iou=0.5):
    """
    使用 YOLO 模型对图像进行目标检测。
    参数:
        chosen_model: YOLO 模型实例。
        img: 输入图像。
        classes: 指定检测的类别列表（可选，默认为空列表）。
        conf: 检测的置信度阈值（默认为 0.5）。
        iou: NMS 的 IoU 阈值（默认为 0.5）。

    返回:
        results: 检测结果。
    """
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, iou=iou)
    else:
        results = chosen_model.predict(img, conf=conf, iou=iou)
    return results

# 定义预测并绘制检测框的函数
def predict_and_detect(chosen_model, img, tracker, classes=[], conf=0.5, iou=0.5, rectangle_thickness=2, text_thickness=1, font_scale=0.5):
    """
    对图像进行目标检测，并在检测到的目标周围绘制矩形框和类别名称及置信度。

    参数:
        chosen_model: YOLO 模型实例。
        img: 输入图像。
        tracker: DeepSort 跟踪器实例。
        classes: 指定检测的类别列表（可选，默认为空列表）。
        conf: 检测的置信度阈值（默认为 0.5）。
        iou: NMS 的 IoU 阈值（默认为 0.5）。
        rectangle_thickness: 矩形框的厚度（默认为 2）。
        text_thickness: 文本的厚度（默认为 1）。
        font_scale: 字体大小（默认为 0.5）。

    返回:
        img: 绘制了检测框和类别名称及置信度的图像。
    """
    results = predict(chosen_model, img, classes, conf=conf, iou=iou)
    detections = []

    # “行人”类别 ID 为 0，“人”类别 ID 为 1
    valid_classes = [0, 1]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框坐标
            conf = float(box.conf[0])  # 置信度
            cls_id = int(box.cls[0])  # 类别 ID
            if cls_id in valid_classes and conf > 0.4:  # 只处理“人”和“行人”类别且置信度 > 0.4
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))  # DeepSort 需要的格式：[x, y, w, h]

    # 使用 DeepSort 更新目标跟踪
    tracks = tracker.update_tracks(detections, frame=img)

    # 绘制检测框和 ID
    for track in tracks:
        if not track.is_confirmed():  # 只处理已确认的目标
            continue
        track_id = track.track_id  # 目标 ID

        # 平滑检测框位置
        smoothed_bbox = smooth_box(track, alpha=0.7)  # 使用平滑函数
        x1, y1, x2, y2 = smoothed_bbox

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), rectangle_thickness)

        # 注释掉绘制目标 ID 的代码
        # cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)

        # 记录已跟踪的 ID
        if track_id not in tracked_ids:
            tracked_ids.add(track_id)
            global person_count
            person_count += 1
            print(f"检测到新的目标! 当前总人数为: {person_count}")  # 打印当前总人数

    # 在图像左上角显示总人数
    cv2.putText(img, f"Total people: {person_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

# 初始化已跟踪的 ID 集合和总人数计数器
tracked_ids = set()
person_count = 0

# 加载 YOLO 模型
model = YOLO('runs/detect/train5/weights/best.pt')

# 定义输入视频路径
video_path = r"video/people3.mp4"

# 解析输入视频路径
video_dir = os.path.dirname(video_path)
video_name = os.path.basename(video_path)
output_filename = f"test_{video_name}"

# 创建 videoout 文件夹（如果不存在）
output_folder = "videoout"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 拼接输出视频的完整路径
output_path = os.path.join(output_folder, output_filename)

# 打开输入视频文件
cap = cv2.VideoCapture(video_path)

# 创建视频写入器
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义视频编码格式
writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 创建指定大小的窗口
cv2.namedWindow("YOLO11 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO11 Detection", 640, 480)

# 循环读取视频帧
while True:
    success, img = cap.read()
    if not success:
        break

    # 进行目标检测和跟踪
    result_img = predict_and_detect(model, img, tracker, classes=[0, 1], conf=0.6, iou=0.6)

    # 写入处理后的帧
    writer.write(result_img)

    # 显示处理后的帧
    cv2.imshow("YOLO11 Detection", result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

# 释放资源
writer.release()
cap.release()
cv2.destroyAllWindows()

# 打印最终统计的人数
print(f"最终统计的人数为: {person_count}")