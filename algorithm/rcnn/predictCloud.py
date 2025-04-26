#----------------------------------------------------#
#   实现视频检测功能，可调用摄像头或者视频进行检测。
#----------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
import ffmpeg

from frcnn import FRCNN

# 定义全局变量 frcnn
frcnn = FRCNN()

def is_target_detected(frame):
    """
    判断帧中是否检测到目标
    :param frame: 输入的帧
    :return: 是否检测到目标的布尔值
    """
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    _, _, classes = frcnn.detect_image(frame, return_classes=True)
    return len(classes) > 0

def get_detected_target_types(frame):
    """
    获取检测到的目标的类型
    :param frame: 输入的帧
    :return: 检测到的目标类型列表
    """
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    _, _, classes = frcnn.detect_image(frame, return_classes=True)
    return classes

# 定义平滑检测框位置的函数
def smooth_box(box, alpha=0.5):
    """
    使用指数加权平均平滑检测框位置。
    参数:
        box: 当前检测框，这里假设 box 是 [left, top, right, bottom] 格式
        alpha: 平滑系数（0 < alpha < 1）。
    返回:
        平滑后的检测框坐标。
    """
    box_id = tuple(box.astype(int))  # 使用框的坐标作为键
    if box_id not in smoothed_boxes:
        smoothed_boxes[box_id] = box  # 初始化为当前框的位置
    else:
        # 使用指数加权平均来平滑位置
        smoothed_boxes[box_id] = alpha * smoothed_boxes[box_id] + (1 - alpha) * box
    return smoothed_boxes[box_id]

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   由于只保留视频模式，这两个参数在当前代码中无实际作用
    #-------------------------------------------------------------------------#
    crop = False
    count = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   output_stream_url   表示输出视频流的RTMP地址
    #----------------------------------------------------------------------------------------------------------#
    video_path = 'rtmp://localhost:1935/flv-live/test'
    output_stream_url = 'rtmp://localhost:1935/flv-live/test02'  # 输出视频流的RTMP地址

    # 初始化平滑检测框的字典
    smoothed_boxes = {}

    capture = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not capture.isOpened():
        print(f"Error opening video file: {video_path}")
        exit()

    # 获取输入视频流的属性
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用FFmpeg启动推流进程
    process = (
        ffmpeg
        .input('pipe:0', format="rawvideo", pix_fmt="bgr24", s=f'{width}x{height}')
        .output(output_stream_url, vcodec='libx264', pix_fmt='yuv420p', r=fps, s=f'{width}x{height}', format='flv')
        .run_async(pipe_stdin=True)
    )

    # 创建一个可调整大小的窗口，并设置初始大小
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", 640, 480)

    fps = 0.0
    while True:
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()

        # 检查是否成功读取帧
        if not ref:
            print("End of video.")
            break

        # 判断是否检测到目标
        detected = is_target_detected(frame, frcnn)
        if detected:
            target_types = get_detected_target_types(frame, frcnn)
            print("检测到的目标类型:", target_types)

        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame_pil = Image.fromarray(np.uint8(frame))
        # 进行检测
        result_img, boxes, classes = frcnn.detect_image(frame_pil)
        result_img = np.array(result_img)

        # 平滑检测框并重新绘制
        for i, box in enumerate(boxes):
            smoothed_box = smooth_box(np.array(box))
            left, top, right, bottom = smoothed_box.astype(int)
            class_name = classes[i]
            cv2.rectangle(result_img, (left, top), (right, bottom), (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(result_img, class_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        # RGBtoBGR满足opencv显示格式
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        result_img = cv2.putText(result_img, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        process.stdin.write(result_img.tobytes())  # 写入 FFmpeg 标准输入流
        cv2.imshow("video", result_img)
        c = cv2.waitKey(1) & 0xff

        if c == 27:
            break

    capture.release()
    process.stdin.close()
    process.wait()
    cv2.destroyAllWindows()