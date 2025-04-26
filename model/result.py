class Result:
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg

    def to_dict(self):
        return {
            'code': self.code,
            'message': self.msg
        }
