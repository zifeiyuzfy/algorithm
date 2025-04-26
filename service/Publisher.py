# 生产者，用于发放frame给各个模型
import cv2


class Publisher:
    def __init__(self, ffmpegProcess):
        self.ModelAlgorithms = []  # 存储所有消费者（各个模型）的队列
        self.ffmpegProcess = ffmpegProcess

    def subscribe(self, modelAlgorithm):
        """订阅者注册"""
        self.ModelAlgorithms.append(modelAlgorithm)

    def publish(self, msg):
        """发布消息到所有订阅者的队列"""
        for modelAlgorithm in self.ModelAlgorithms:
            modelAlgorithm.inputFrame(msg)

    def receive(self):
        # 后续考虑拼接
        # for modelAlgorithm in self.ModelAlgorithms:
        #     modelAlgorithm.getOutPutFrame()
        while True:
            processedFrame = self.ModelAlgorithms[0].getOutPutFrame()
            print("receive")
            # 将处理后的帧传给 FFmpeg 进行推流
            self.ffmpegProcess.stdin.write(processedFrame.tobytes())  # 写入 FFmpeg 标准输入流

            # 显示当前帧（可选）
            # cv2.imshow('Stream', processedFrame)

            # 按 'q' 键退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
