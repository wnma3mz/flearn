from flask import Flask, jsonify, make_response, request

from flearn.client import Client
from flearn.client.utils import load_client_conf


# add route in class
class EndpointAction(object):
    """Flask添加路由"""

    def __init__(self, action):
        self.action = action

    def __call__(self, *args):
        # Perform the action
        answer = self.action()
        # Send it
        return jsonify(answer)


class Communicator(object):
    app = None

    def __init__(self, **conf):
        """通信模块.

        使用Flask进行HTTP通信

        Args:
            conf (dict): {
                "model" :        torchvision.models
                                 模型,

                "criterion" :    torch.nn.modules.loss
                                 损失函数,

                "optimizer" :    torch.optim
                                 优化器,

                "trainloader" :  torch.utils.data
                                 训练数据集,

                "testloader" :   torch.utils.data
                                 测试数据集

                "fpath" :        str
                                 客户端配置文件路径名称
            }
            客户端设置参数
        """
        self.conf = load_client_conf(**conf)

        self.app = Flask(__name__)
        self.app.add_url_rule(
            "/train", "/train", EndpointAction(self.client_train), methods=["POST"]
        )
        self.app.add_url_rule(
            "/upload", "/upload", EndpointAction(self.client_upload), methods=["POST"]
        )
        self.app.add_url_rule(
            "/revice", "/revice", EndpointAction(self.client_revice), methods=["POST"]
        )
        self.app.add_url_rule(
            "/evaluate",
            "/evaluate",
            EndpointAction(self.client_evaluate),
            methods=["POST"],
        )

    def client_train(self):
        """训练模型指令.

        Returns:
            dict: Dict {
                "code" :      int
                              状态码

                "msg" :       str
                              状态消息

                "loss" :      float
                              损失值

                "train_acc" : float
                              模型在训练集上的精度
            }
        """
        i = request.json["round"]
        train_json = self.client_model.train(i)
        return train_json

    def client_upload(self):
        """上传模型参数.

        Returns:
            dict: Dict {
                "datas" : str
                          经过编码（加密）后的模型字符串

                "fname" : str
                          模型名称

                "round" : int or str
                          第i轮模型

                "msg" :   str
                          状态消息
            }
        """
        i = request.json["round"]
        upload_json = self.client_model.upload(i)
        return upload_json

    def client_revice(self):
        """接收模型参数.

        Returns:
            dict: Dict {
                "code" :     int
                             状态码

                "msg" :      str
                             状态消息
            }
        """
        r = request.json
        i = r["round"]
        w_glob_b64_str = r["glob_params"]
        revice_json = self.client_model.revice(i, w_glob_b64_str)
        return revice_json

    def client_evaluate(self):
        """评估模型指令.

        Returns:
            dict: Dict {
                "code" :      int
                              状态码

                "msg" :       str
                              状态消息

                "test_acc" :  float
                              模型在训练集上的精度
            }
        """
        i = request.json["round"]
        evaluate_json = self.client_model.evaluate(i)
        return evaluate_json

    def run(self, client_model=Client, port=6000):
        """启动Flask HTTP.

        Args:
            CustomClient : object
                           自定义客户端对象

            port :         int
                           HTTP端口号
        """
        self.client_model = client_model(self.conf)
        self.app.run(host="0.0.0.0", port=port, debug=True)
