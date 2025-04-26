from algorithm.yolo.testvideo import predict
from algorithmService.ModelAlgorithm import ModelAlgorithm


class YoloModelAlgorithm(ModelAlgorithm):
    def consume(self):
        """阻塞式消费消息"""
        while True:
            # 从队列中获取输入帧
            frame = self.inputFrameQueue.get()
            # 经过模型处理完后输出到结果队列中
            self.outputFrameQueue.put(predict(frame))
