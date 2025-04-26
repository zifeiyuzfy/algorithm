from queue import Queue
from abc import ABC, abstractmethod


# 模型处理抽象类
class ModelAlgorithm(ABC):
    def __init__(self, name, publisher=None):
        self.name = name
        self.inputFrameQueue = Queue()
        self.outputFrameQueue = Queue()
        # 绑定生产者
        if publisher is not None:
            self.publisher = publisher
            self.publisher.subscribe(self)

    @abstractmethod
    def consume(self):
        pass

    def inputFrame(self, frame):
        self.inputFrameQueue.put(frame)

    def getOutPutFrame(self):
        return self.outputFrameQueue.get()
