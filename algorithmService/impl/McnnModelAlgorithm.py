import json

import requests

from algorithm.mcnn.testvideo import estimate_density_map
from algorithmService.ModelAlgorithm import ModelAlgorithm
from utils import jsonUtil


class McnnModelAlgorithm(ModelAlgorithm):
    def consume(self):
        """阻塞式消费消息"""
        while True:
            # 从队列中获取输入帧
            frame = self.inputFrameQueue.get()
            number = estimate_density_map(frame)
            url = "http://localhost:7001/publish"
            data = {
                "topic": "1/3",
                "message": number
            }
            header = {
                "Authorization": "Bearer MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAkXkJNcCBcyuo763+fPt5",
                "Content-Type": "application/json"
            }
            print("target_types", json.dumps(data))
            res = requests.post(url, json.dumps(data), headers=header)
            print("发送消息成功", res)
