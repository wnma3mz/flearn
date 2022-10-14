from flask import Flask, jsonify, make_response, request


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

    def __init__(self, client):
        """通信模块.

        使用Flask进行HTTP通信

        Args:
            Client :       object
                           Flearn Client

            客户端设置参数
        """
        self.client = client

        self.app = Flask(__name__)
        self.app.add_url_rule("/train", "/train", EndpointAction(self.client_train), methods=["POST"])
        self.app.add_url_rule("/upload", "/upload", EndpointAction(self.client_upload), methods=["POST"])
        self.app.add_url_rule("/receive", "/receive", EndpointAction(self.client_receive), methods=["POST"])
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
        train_json = self.client.train(i)
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
        upload_json = self.client.upload(i)
        return upload_json

    def client_receive(self):
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
        receive_json = self.client.receive(i, w_glob_b64_str)
        return receive_json

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
        evaluate_json = self.client.evaluate(i)
        return evaluate_json

    def run(self, port=6000):
        """启动Flask HTTP.

        Args:
            port :         int
                           HTTP端口号
        """
        self.app.run(host="0.0.0.0", port=port, debug=True)
