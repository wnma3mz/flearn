# coding: utf-8
import base64
import pickle

import numpy as np
from flearn.common import Trainer, init_strategy


class Server(object):
    def __init__(self, conf, **strategy_p):
        """服务端

        Args:
            conf (dict): {
                    "model_fpath" :     str
                                        模型存储路径

                    "client_numbers":   int
                                        客户端总数

                    "strategy_name":    str
                                        联邦学习策略名称

                    "strategy":         Strategy, default: None
                                        联邦学习策略

                    "eval_conf":        dict, default: None
                                        测试模型配置参数。
                                        {
                                            "model":        全局模型结构,

                                            "criterion":    损失函数,

                                            "device":       gpu or cpu,

                                            "display":      False,

                                            "dataloader":   测试数据集,

                                            "eval_clients": False, 是否测试客户端
                                        }
            } 服务端配置文件

            strategy_p: dict
                        联邦学习策略配置文件。
                        {"shared_key_layers": 共享的参数名称}
        """
        listed_keys = [
            "model_fpath",
            "client_numbers",
            "strategy_name",
            "strategy",
            "eval_conf",
        ]
        # 设置属性，两种方法均可
        for k in conf.keys():
            if k in listed_keys:
                self.__dict__[k] = conf[k]
                # self.__setattr__(k, kwargs[k])

        self.strategy = conf["strategy"] if "strategy" in conf.keys() else None
        self.eval_conf = conf["eval_conf"] if "eval_conf" in conf.keys() else None

        strategy_p["model_fpath"] = self.model_fpath
        if self.strategy == None:
            self.strategy = init_strategy(self.strategy_name, **strategy_p)

    @staticmethod
    def active_client(lst, k):
        if k != -1:
            k = min(len(lst), k)
            return np.random.choice(lst, size=k, replace=False)
        return lst

    @staticmethod
    def mean_lst(k, lst):
        return np.mean(list(map(lambda x: x[k], lst)))

    def drop_client(self, val_acc_lst):
        print("精度: {}".format(val_acc_lst))
        # cifar10: 1 / 10 * 100 * 1.2 = 12
        idx_lst = [idx for idx, val_acc in enumerate(val_acc_lst) if val_acc > 12]
        # idx_lst = [np.argmax(val_acc_lst)]
        if len(idx_lst) == 0:
            return []
        print("精度超过12%的客户端索引id: {}".format(idx_lst))
        return idx_lst

    def train(self, data_lst):
        loss = self.mean_lst("loss", data_lst)
        train_acc = self.mean_lst("train_acc", data_lst)
        # 如果存在验证集->drop-worst
        val_acc_lst = list(map(lambda x: x["val_acc"], data_lst))
        if val_acc_lst[0] == -1:
            return loss, train_acc, range(len(data_lst))
        else:
            idx_lst = self.drop_client(val_acc_lst)
            return loss, train_acc, idx_lst

    def upload(self):
        pass

    def ensemble(self, data_lst, round_, k=-1, **kwargs):
        """服务端的集成部分

        Args:
            data_lst :      list
                            各个客户端发送过来的模型参数

            round_ :        str
                            第x轮

            k :             int
                            选取k个客户端进行聚合

            kwargs :        dict
                            集成策略需要的额外参数

        Returns:
            dict : Dict {
                "glob_params" : str
                                编码后的全局模型

                "code" :        int
                                状态码,

                "msg" :         str
                                状态消息,
            }
        """
        ensemble_params_lst, client_id_lst = [], []
        for item in data_lst:
            if int(round_) != int(item["round"]):
                continue
            model_params_encode = item["datas"].encode()
            model_params_b = base64.b64decode(model_params_encode)
            model_data = pickle.loads(model_params_b)

            client_id_lst.append(item["client_id"])
            ensemble_params_lst.append(model_data)

            # 存储发送过来的参数
            # with open(os.path.join(self.model_fpath, item["fname"]), "wb") as f:
            #     f.write(model_params_b)
        # 做进一步筛选，比如按照agg_weight排序
        # if k != 0:
        #     idxs_users = sorted(range(N), key=lambda x: agg_weight_lst[x])[:m]

        return self.strategy.server(ensemble_params_lst, round_)

    def revice(self):
        pass

    def evaluate(self, data_lst, is_select=False):
        """服务端的评估部分

        Args:
            data_lst :      list
                            客户端的id列表 or 客户端测试精度

            is_select :     bool
                            是否为选择客户端id

        Returns:
            list or str:
        """

        if is_select == True:
            # 如果需要在服务器端进行模型测试并且仍旧需要在客户端测试
            if self.eval_conf != None:
                return data_lst if self.eval_conf["eval_clients"] == True else []
            else:
                return data_lst

        def process_acc_str(data_lst):
            print(data_lst)
            test_acc_lst = np.mean(list(map(lambda x: x["test_acc"], data_lst)), axis=0)
            print(test_acc_lst)
            return "; ".join("{:.4f}".format(x) for x in test_acc_lst)

        if self.eval_conf == None:
            test_acc = process_acc_str(data_lst)
        else:
            _, test_acc_server = self.server_eval()
            test_acc = "[Server] {:.4f}".format(test_acc_server)
            # 如果仍需要在客户端进行测试的情况
            if self.eval_conf["eval_clients"] == True:
                test_acc += "; [Clients] {}".format(process_acc_str(data_lst))
        return test_acc

    def server_eval(self):
        """在服务器端对模型进行评估

        Returns:
            tuple:
                loss, acc
        """
        trainer = Trainer(
            self.eval_conf["model"],
            None,
            self.eval_conf["criterion"],
            self.eval_conf["device"],
            self.eval_conf["display"],
        )

        return trainer.test(self.eval_conf["dataloader"])
