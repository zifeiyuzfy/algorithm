import json

import redis

from utils.jsonUtil import json_loads

# 创建 Redis 连接
r = redis.Redis(
    host='localhost',  # Redis 服务器地址
    port=6379,  # Redis 服务器端口
    decode_responses=True  # 自动解码字节为字符串
)
# 测试连接
try:
    r.ping()
    print("成功连接到Redis")
except Exception as e:
    print("连接Redis失败:", e)
    exit()


def getRedisClient():
    return r

