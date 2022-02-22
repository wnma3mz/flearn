# coding: utf-8
import base64
import pickle


class Encrypt(object):
    """通信加密"""

    def __init__(self):
        pass

    def encode(self, params):
        """模型编码

        Args:
            params ():   客户端模型参数

        Returns:
            str: 编码后的模型参数
        """
        # 加密参数为字符串
        model_parambs_b64 = base64.b64encode(pickle.dumps(params))
        model_b64_str = model_parambs_b64.decode()

        return model_b64_str

    def decode(self, glob_params):
        """模型解码

        Args:
            glob_params (str):   全局参数（服务端传回的模型参数）

        Returns:
            array: 解码后的模型参数
        """
        # Client-side copy
        w_glob_encode = glob_params.encode()
        w_glob_b = base64.b64decode(w_glob_encode)
        return pickle.loads(w_glob_b)


class EmptyEncrypt(object):
    def encode(self, params):
        return params

    def decode(self, glob_params):
        return glob_params
