import json

"""
为了适配Java的JSON序列化这里需要进行两次JSON编解码
"""


def json_loads(data):
    try:
        # 尝试直接解析
        result = json.loads(data)
        # 如果解析结果是字符串，尝试递归解析
        if isinstance(result, str):
            return json_loads(result)
        return result
    except json.JSONDecodeError:
        # 如果解析失败，返回原始数据
        return data


def json_dumps(obj):
    return json.dumps(json.dumps(obj))
