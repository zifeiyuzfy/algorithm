from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def process(self, frame):
        """子类必须实现的抽象方法"""
        pass
