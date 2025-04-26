import json

import requests

from algorithm.ssd.predict import is_target_detected, get_detected_target_types
from algorithmService.ModelAlgorithm import ModelAlgorithm
from utils import jsonUtil


class SsdModelAlgorithm(ModelAlgorithm):
    def consume(self):
        """阻塞式消费消息"""
        while True:
            # 从队列中获取输入帧
            frame = self.inputFrameQueue.get()
            detected = is_target_detected(frame)
            if detected:
                target_types = list(get_detected_target_types(frame))
                url = "http://localhost:7001/publish"
                data = {
                    "topic": "1/2",
                    "message": str(target_types)
                }
                header = {
                    "Authorization": "Bearer MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAkXkJNcCBcyuo763+fPt5",
                    "Content-Type": "application/json"
                }
                print("target_types", json.dumps(data))
                res = requests.post(url, json.dumps(data), headers=header)
                print("发送消息成功", res)
