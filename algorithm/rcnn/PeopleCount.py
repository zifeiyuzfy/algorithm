# ----------------------------------------------------#
#   将单张图片预测、摄像头检测
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time
import cv2
import numpy as np
import os  # 新增导入
from PIL import Image
from frcnn import FRCNN

# 定义全局变量 frcnn
frcnn = FRCNN()


def _process_frame_for_detection(frame):
    """
    内部方法：处理帧用于检测
    :param frame: 输入的帧
    :return: PIL.Image格式的图像
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(frame_rgb))


def get_detection_info(frame):
    """
    获取检测结果和类别计数
    :param frame: 输入的帧
    :return: (检测图像, 类别计数字典)
    """
    pil_image = _process_frame_for_detection(frame)
    detected_image, _, classes = frcnn.detect_image(pil_image, return_classes=True)

    # 统计类别数量
    count_dict = {}
    for cls in classes:
        count_dict[cls] = count_dict.get(cls, 0) + 1

    # 转换回OpenCV格式
    detected_image = cv2.cvtColor(np.array(detected_image), cv2.COLOR_RGB2BGR)
    return detected_image, count_dict


def is_target_detected(frame):
    """
    判断帧中是否检测到目标
    :param frame: 输入的帧
    :return: 是否检测到目标的布尔值
    """
    pil_image = _process_frame_for_detection(frame)
    _, _, classes = frcnn.detect_image(pil_image, return_classes=True)
    return len(classes) > 0


def get_detected_target_types(frame):
    """
    获取检测到的目标的类型
    :param frame: 输入的帧
    :return: 检测到的目标类型列表
    """
    pil_image = _process_frame_for_detection(frame)
    _, _, classes = frcnn.detect_image(pil_image, return_classes=True)
    return classes


def get_target_count(frame, target_classes=None):
    """
    获取指定目标类别的数量
    :param frame: 输入的帧
    :param target_classes: 指定的目标类别列表，如果为None则返回所有类别
    :return: 字典{类别: 数量}
    """
    pil_image = _process_frame_for_detection(frame)
    _, _, classes = frcnn.detect_image(pil_image, return_classes=True)

    count_dict = {}
    for cls in classes:
        count_dict[cls] = count_dict.get(cls, 0) + 1

    if target_classes is not None:
        # 只返回指定的类别
        return {cls: count_dict.get(cls, 0) for cls in target_classes}
    else:
        # 返回所有类别
        return count_dict


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'    单张图片预测
    #   'video'      视频检测
    # ----------------------------------------------------------------------------------------------------------#
    mode = "video"
    # -------------------------------------------------------------------------#
    #   crop        是否截取目标
    #   count       是否显示计数（已整合到检测流程）
    # -------------------------------------------------------------------------#
    crop = False
    count = True  # 始终启用计数功能

    # 图片预测模式
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                # 检测并获取结果
                detected_image, count_dict = get_detection_info(np.array(image))

                # 计算总人数
                pedestrian_count = count_dict.get('pedestrian', 0)
                people_count = count_dict.get('people', 0)
                total_people = pedestrian_count + people_count

                # 在图像上绘制总人数
                cv2.putText(detected_image, f"Total People: {total_people}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 保存图片到 img_out 文件夹，命名为 test_原文件名
                output_folder = 'img_out'
                os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在
                filename = os.path.basename(img)  # 获取图片文件名
                output_path = os.path.join(output_folder, f'test_{filename}')
                cv2.imwrite(output_path, detected_image)

                # cv2.imshow("Prediction", detected_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 视频检测模式
    elif mode == "video":
        video_path = "video/people2.mp4"
        video_save_path = "video_out/people2_out.mp4"

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f"Error opening video source: {video_path}")
            exit()

        # 视频写入设置
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, 25, size)

        cv2.namedWindow("Faster R-CNN Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Faster R-CNN Detection", 640, 480)

        fps = 0.0
        try:
            while True:
                t1 = time.time()
                ref, frame = capture.read()
                if not ref:
                    break

                # 获取检测结果和计数
                processed_frame, count_dict = get_detection_info(frame)

                # 计算总人数
                pedestrian_count = count_dict.get('pedestrian', 0)
                people_count = count_dict.get('people', 0)
                total_people = pedestrian_count + people_count

                # 绘制FPS和总人数信息
                fps = (fps + (1. / (time.time() - t1))) / 2
                cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Total people: {total_people}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Faster R-CNN Detection", processed_frame)
                if video_save_path != "":
                    out.write(processed_frame)

                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
        finally:
            capture.release()
            if video_save_path != "":
                out.release()
            cv2.destroyAllWindows()

    else:
        raise ValueError("Invalid mode! Use 'predict' or 'video'")