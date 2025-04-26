# -----------------------------------------------------------------------#
#   简化版SSD检测，仅支持图片预测和视频检测两种模式
#   专注于行人(pedestrian)和人群(people)计数功能
# -----------------------------------------------------------------------#
import time
import cv2
import numpy as np
import os  # 新增导入
from PIL import Image
from ssd import SSD

# 初始化SSD检测器
ssd = SSD()


def _process_frame(frame):
    """统一帧处理函数"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(frame_rgb))


def detect_and_count(frame):
    """
    检测并计数
    :param frame: 输入帧
    :return: (处理后的图像, 计数字典)
    """
    pil_image = _process_frame(frame)
    detected_image, classes = ssd.detect_image(pil_image, return_classes=True)

    # 统计目标数量
    counts = {
        'pedestrian': 0,
        'people': 0
    }

    for cls in classes:
        if cls == 'pedestrian':
            counts['pedestrian'] += 1
        elif cls == 'people':
            counts['people'] += 1

    # 转换图像格式
    detected_image = cv2.cvtColor(np.array(detected_image), cv2.COLOR_RGB2BGR)
    return detected_image, counts


if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   模式选择
    #   'predict' - 单张图片预测模式
    #   'video'   - 视频检测模式
    # -------------------------------------------------------------------------#
    # mode = "predict"
    mode = "video"
    if mode == "predict":
        # 图片预测模式
        while True:
            img_path = input('输入图片路径(输入q退出):')
            if img_path.lower() == 'q':
                break

            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    raise ValueError("无法读取图片")

                # 检测并计数
                result_img, counts = detect_and_count(frame)

                # 计算总人数
                total_people = counts['pedestrian'] + counts['people']

                # 显示总人数
                cv2.putText(result_img, f"Total people: {total_people}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 保存图片到 img_out 文件夹，命名为 test_原文件名
                output_folder = 'img_out'
                os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在
                filename = os.path.basename(img_path)  # 获取图片文件名
                output_path = os.path.join(output_folder, f'test_{filename}')
                cv2.imwrite(output_path, result_img)

                # cv2.imshow("Detection Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"错误: {e}")

    elif mode == "video":
        # 视频检测模式
        video_path = "video/people2.mp4"
        output_path = "video_out/people2_out.mp4"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            exit()

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 初始化视频写入器
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))

        cv2.namedWindow("SSD Detection", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()

                # 检测并计数
                result_img, counts = detect_and_count(frame)

                # 计算总人数
                total_people = counts['pedestrian'] + counts['people']

                # 计算FPS
                current_fps = 1.0 / (time.time() - start_time)

                # 显示信息
                cv2.putText(result_img, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_img, f"Total people: {total_people}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("SSD Detection", result_img)
                writer.write(result_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            print("视频处理完成")

    else:
        print("错误: 请选择'predict'或'video'模式")