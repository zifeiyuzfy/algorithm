import threading
import time
import cv2
import ffmpeg
from algorithmService.impl.RcnnModelAlgorithm import RcnnModelAlgorithm
from algorithmService.impl.YoloModelAlgorithm import YoloModelAlgorithm
from constant.deviceStreamMap import DEVICE_STREAM_MAP
from globals.globalEvent import step_completed_event
from service.Publisher import Publisher


# 将每一帧数据传给算法处理
# def model_process(frame, model_type):
#     match model_type:
#         case 1:
#             return process(frame)


class StreamService:
    def __init__(self, deviceId):
        self.firstFlag = True
        self.running = True
        self.deviceId = deviceId

    # 处理流媒体数据
    def processStream(self, input_url, output_url, model_type):
        # 打开输入输出流
        # 打开RTMP输入流
        cap = cv2.VideoCapture(input_url)
        while not cap.isOpened():
            cap = cv2.VideoCapture(input_url)
            print("can not open stream")
            time.sleep(1)

        # 获取视频的基本信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 使用 FFmpeg 启动推流进程（保持进程持续运行）
        ffmpegProcess = (
            ffmpeg
            .input('pipe:0', format="rawvideo", pix_fmt="bgr24", s=f'{width}x{height}')  # 从标准输入流读取帧
            .output(output_url, vcodec='libx264', pix_fmt='yuv420p', r=fps, s=f'{width}x{height}',
                    format='flv')  # 输出到 RTMP 流
            .run_async(pipe_stdin=True)  # 异步执行 FFmpeg，允许我们向标准输入写入数据
        )

        publisher = Publisher(ffmpegProcess)

        # 创建多个消费者并订阅
        match model_type:
            case 1:
                YoloModelAlgorithm("Yolo", publisher)

        # 启动消费者线程
        threads = []
        for modelAlgorithm in publisher.ModelAlgorithms:
            t = threading.Thread(target=modelAlgorithm.consume)
            t.start()
            threads.append(t)
        threading.Thread(target=publisher.receive).start()

        # 创建后台检测算法
        rcnnModelAlgorithm = RcnnModelAlgorithm("Rcnn")
        threading.Thread(target=rcnnModelAlgorithm.consume).start()
        cnt = 0  # 记录帧数。每10帧传1帧给后台算法检测
        print("启动")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧，退出")
                break

            # 将每一帧传给模型处理
            # processedFrame = model_process(frame, model_type)
            cnt = cnt + 1
            if cnt % 100 == 0:
                print("进入检测")
                # 将帧发送给后台算法进行检测
                rcnnModelAlgorithm.inputFrame(frame)
                cnt = 0
            # 每一帧发给实时算法检测
            publisher.publish(frame)

            if self.firstFlag:
                step_completed_event.set()
                self.firstFlag = False
            # 将处理后的帧传给 FFmpeg 进行推流
            # ffmpegProcess.stdin.write(processedFrame.tobytes())  # 写入 FFmpeg 标准输入流

            # 显示当前帧（可选）
            # cv2.imshow('Stream', processedFrame)

            # 按 'q' 键退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        # 从Map中删除
        DEVICE_STREAM_MAP.popitem(self.deviceId)
