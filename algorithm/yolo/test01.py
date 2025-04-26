import cv2
import ffmpeg
from ultralytics import YOLO

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
        conf: 检测的置信度阈值（默认为0.5）。
        iou: NMS的IoU阈值（默认为0.5）。
        rectangle_thickness: 矩形框的厚度（默认为4）。
        text_thickness: 文本的厚度（默认为2）。
        font_scale: 字体大小（默认为1.5）。
    返回:
        img: 绘制了检测框和类别名称及置信度的图像。
    """
    results = chosen_model.predict(img, classes=classes, conf=conf, iou=iou)  # 调用预测函数获取检测结果
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
    return img

# 初始化平滑检测框的字典
smoothed_boxes = {}

# 加载YOLO模型
model = YOLO('runs/detect/train4/weights/best.pt')  # 加载训练好的YOLO模型权重文件

# 输入和输出视频流的RTMP地址
input_stream_url = 'rtmp://localhost:1935/flv-live/test'  # 输入视频流的RTMP地址
output_stream_url = 'rtmp://localhost:1935/flv-live/test02'  # 输出视频流的RTMP地址

# 打开输入视频流
cap = cv2.VideoCapture(input_stream_url)
if not cap.isOpened():
    print("无法打开输入视频流")
    exit()

# 获取输入视频流的属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 使用FFmpeg启动推流进程
process = (
    ffmpeg
    .input('pipe:0', format="rawvideo", pix_fmt="bgr24", s=f'{width}x{height}')
    .output(output_stream_url, vcodec='libx264', pix_fmt='yuv420p', r=fps, s=f'{width}x{height}', format='flv')
    .run_async(pipe_stdin=True)
)

print("开始处理视频流...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，退出")
        break

    # 对当前帧进行目标检测并绘制检测框
    processed_frame = predict_and_detect(model, frame)

    # 将处理后的帧传给FFmpeg进行推流
    process.stdin.write(processed_frame.tobytes())

    # # 显示当前帧（可选）
    cv2.imshow('Processed Stream', processed_frame)
    #
    # # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
process.stdin.close()
process.wait()