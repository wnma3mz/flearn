# coding: utf-8
import concurrent.futures
import copy
import threading

import requests

from flearn.common import Logger


class Communicator(object):
    def __init__(self, conf):
        """服务端的通信模块，用于发送指令

        Args:
            conf_fpath :     str
                                服务端配置文件路径

            conf       :     dict
                                服务端配置字典，{

                                    "Round":            200,
                                                        训练总轮数

                                    "client_numbers":   1,
                                                        客户端数量

                                    "dataset_name":     "faces",

                                    "log_name_fmt":     "",

                                    "log":              True,

                                    "log_suffix" :      str
                                                        log名称的后缀, ""

                                    "client_url_lst":   ["http://127.0.0.1:6000/{}"]
                                                        请求客户端的链接，单机情况可换为"client_lst", 优先为该情况

                                    "client_lst" :      [Client]
                                                        训练客户端的对象，多机情况可换为"client_url_lst"
                                }
        """
        self.server = conf["server"]

        # 训练客户端配置
        if "client_url_lst" in conf.keys():
            # 网络请求，对客户端进行请求的API
            self.client_url_lst = conf["client_url_lst"]
            self.thread_func = self.thread_request  # 多线程并行训练
            self.client_id_lst = range(len(self.client_url_lst))  # 为每个客户端分配一个id
        elif "client_lst" in conf.keys():
            # 如果是单机训练
            self.client_url_lst = conf["client_lst"]  # 更换为客户端的对象
            self.thread_func = self.thread_run  # 更换请求函数
            # 根据客户端的client_id重新定义，有待更新
            self.client_id_lst = [x.client_id for x in self.client_url_lst]
        else:
            raise SyntaxError("Please input client_url_lst or client_lst")

        if "client_numbers" in conf.keys():
            client_numbers = conf["client_numbers"]
        else:
            client_numbers = len(self.client_url_lst)

        assert client_numbers == len(self.client_url_lst)

        # 日志相关配置
        self.log = False if "log" in conf.keys() and conf["log"] == False else True
        if self.log:
            log_suffix = conf["log_suffix"] if "log_suffix" in conf.keys() else ""

            if "log_name_fmt" not in conf.keys():
                log_name_fmt = "[Server]{}_round{}_clients{}_{}{}.log"
            else:
                log_name_fmt = conf["log_name_fmt"]

            log_server_name = log_name_fmt.format(
                self.server.strategy_name,
                conf["Round"],
                client_numbers,
                conf["dataset_name"],
                log_suffix,
            )
            self.log_server = Logger(log_server_name, level="info")
            self.log_fmt = (
                "Id: Server; Round: {}; Loss: {:.4f}; TrainAcc: {:.4f}; TestAcc: {};"
            )

        self.active_client_id_lst = copy.deepcopy(self.client_id_lst)  # 每轮选中的客户端进行训练、上传

        # 多线程并发数
        self.max_workers = None

    @staticmethod
    def thread_request(url, command, json_d):
        r = requests.post(url.format(command), json=json_d).json()
        print(r["msg"])
        return r

    @staticmethod
    def thread_run(client, command, json_d):
        if command == "train":
            return client.train(json_d["round"])
        elif command == "upload":
            return client.upload(json_d["round"])
        elif command == "revice":
            return client.revice(json_d["round"], json_d["glob_params"])
        elif command == "evaluate":
            return client.evaluate(json_d["round"])
        else:
            raise SystemError(
                "command must in ['train', 'upload', 'revice', 'evaluate']"
            )

    def get_data_lst(self, command, json_d):
        """服务端发送指令

        Args:
            command :  str
                       assert command in ['train', 'upload', 'revice']

            json_d :   dict
                       发送给客户端的参数{'round': ri} or 含全局参数

        Returns:
            list :     data_lst
                       客户端返回信息组成的list
        """
        data_lst = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_url = {
                executor.submit(self.thread_func, url, command, json_d): url
                for idx, url in zip(self.client_id_lst, self.client_url_lst)
                if idx in self.active_client_id_lst
            }
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    data_lst.append(data)
                except Exception as exc:
                    print("%r generated an exception: %s" % (url, exc))
                    raise SystemError("Please check {}".format(command))
        return data_lst

    def run(self, ri, k=-1, print_round=1, **kwargs):
        """服务端发送指令，控制流主函数

        Args:
            ri :            int or str
                            整个过程中的第x轮

            k  :            int, default: -1
                            每轮选择多少个客户端训练并上传，默认为-1，全部训练并上传

            print_round :   int, default: 1
                            间隔多少轮进行一次评估，默认为1轮


        Returns:
            float :    loss
                       客户端平均损失值

            float :    train_acc
                       客户端平均训练精度

            float :    test_acc
                       客户端平均测试精度

        Note:
            以联邦学习每一轮为最小单位，按顺序分别执行如下步骤：
                1. 选择客户端进行训练
                2. 发送上传参数指令，服务器端接收参数
                3. 服务器端根据策略进行聚合
                4. 发送接收（下载）参数指令，客户端接收更新后的参数
                5. 发送评估指令，客户端在测试集上进行测试
        """
        json_d = {"round": ri}
        # 选择客户端训练(随机)
        self.active_client_id_lst = self.server.active_client(self.client_id_lst, k)
        # 发送训练指令
        data_lst = self.get_data_lst("train", json_d)
        loss, train_acc, id_lst = self.server.train(data_lst)
        log_msg = ri, loss, train_acc, ""
        if id_lst == []:
            if self.log:
                self.log_server.logger.info(self.log_fmt.format(*log_msg))
            self.active_client_id_lst = self.client_id_lst
            return loss, train_acc, ""
        self.active_client_id_lst = [self.active_client_id_lst[idx] for idx in id_lst]

        # 选择客户端上传(随机)
        # self.active_client_id_lst = self.server.active_client(self.client_id_lst, k)
        # print(self.active_client_id_lst)

        # 发送上传指令
        data_lst = self.get_data_lst("upload", json_d)

        # 聚合
        json_d["glob_params"] = self.server.ensemble(data_lst, ri, k=k, **kwargs)
        if json_d["glob_params"] == "":
            raise SystemError("Aggregation error!")

        # 发送参数，客户端接收
        self.active_client_id_lst = self.client_id_lst
        _ = self.get_data_lst("revice", json_d)

        test_acc = ""
        # 为避免测试时间过久，间隔x轮进行测试并输出
        if (ri + 1) % print_round == 0:
            # 评估客户端
            self.active_client_id_lst = self.server.evaluate(
                self.client_id_lst, is_select=True
            )
            data_lst = self.get_data_lst("evaluate", {"round": ri})
            test_acc = self.server.evaluate(data_lst)

        self.active_client_id_lst = self.client_id_lst
        log_msg = ri, loss, train_acc, test_acc
        if self.log:
            self.log_server.logger.info(self.log_fmt.format(*log_msg))
        return loss, train_acc, test_acc
