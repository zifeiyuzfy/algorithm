import json
import threading
import time

import requests
from flask import Flask, jsonify
from flask_cors import CORS

from constant.deviceStreamMap import DEVICE_STREAM_MAP
from globals.globalEvent import step_completed_event
from constant.streamConstant import RTMP_URL_PREFIX, REDIS_MSDKCONNECT_KEY, UAV_RTMP_URL_PREFIX
from model.result import Result
from service.StreamService import StreamService
from utils.jsonUtil import json_loads, json_dumps
from utils.redisUtil import getRedisClient

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():  # put application's code here
    return jsonify(Result(200, 'success').to_dict())


@app.route('/startAi/<deviceId>')
def startAi(deviceId):
    # 视频流地址
    input_url = UAV_RTMP_URL_PREFIX + deviceId

    redisClient = getRedisClient()
    hash_key = REDIS_MSDKCONNECT_KEY
    deviceString = redisClient.hget(hash_key, deviceId)
    device = json_loads(deviceString)
    algorithmId = device['algorithmId']
    if algorithmId == 0:
        # 不需要算法，直接将状态设为ready并返回
        device['isVideoReady'] = True
        redisClient.hset(hash_key, deviceId, json_dumps(device))
        return jsonify(Result(201, '不需要挂载算法').to_dict())
    # 拼接输出流地址
    output_url = f"{RTMP_URL_PREFIX}algorithm{algorithmId}-{deviceId}"
    streamService = StreamService(deviceId)
    threading.Thread(target=streamService.processStream, args=(input_url, output_url, algorithmId)).start()
    # 将处理类放入Map中后续才可以控制退出
    DEVICE_STREAM_MAP[deviceId] = streamService
    # 等待processStream方法获取处理完第一帧
    step_completed_event.wait()
    device['isVideoReady'] = True
    redisClient.hset(hash_key, deviceId, json_dumps(device))
    return jsonify(Result(200, 'success').to_dict())


@app.route('/stopAi/<deviceId>')
def stopAi(deviceId):
    if deviceId not in DEVICE_STREAM_MAP:
        return jsonify(Result(201, 'device not started').to_dict())
    # 退出处理
    DEVICE_STREAM_MAP[deviceId].running = False
    return jsonify(Result(200, 'success').to_dict())


# nacos服务
def service_register():
    url = "http://localhost:8848/nacos/v1/ns/instance?serviceName=algorithm-service&ip=127.0.0.1&port=5001"
    res = requests.post(url)

    print("向nacos注册中心，发起服务注册请求，注册响应状态： {}".format(res.status_code))


# 服务检测
def service_beat():
    while True:
        url = "http://localhost:8848/nacos/v1/ns/instance/beat?serviceName=algorithm-service&ip=127.0.0.1&port=5001"
        res = requests.put(url)
        print("已注册服务，执行心跳服务，续期服务响应状态： {}".format(res.status_code))
        time.sleep(5)


if __name__ == '__main__':
    # service_register()
    # 5秒以后，异步执行service_beat()方法
    # threading.Timer(5, service_beat).start()
    app.run(port=5001, debug=True)
