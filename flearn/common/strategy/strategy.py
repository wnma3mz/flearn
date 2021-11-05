# coding: utf-8
import os
import pickle
from abc import ABC, abstractmethod
from functools import reduce

import numpy as np
import torch
from flearn.common.encrypt import Encrypt


class Strategy(ABC):
    """联邦学习策略的基类，包含对客户端模型处理、服务端聚合等"""

    def __init__(self, model_fpath):
        """

        Args:
            model_fpath : str
                          模型存储的路径
        """
        self.model_fpath = model_fpath
        self.encrypt = Encrypt()

    def server_pre_processing(self, ensemble_params_lst):
        """提取服务器端收到的的参数

        Args:
            ensemble_params_lst :   list
                                    每个客户端发送的参数组成的列表

        Returns:
            tuple : (
                list            : 每个模型分配的聚合权重

                list            : 每个模型的参数
            )
        """
        agg_weight_lst, w_local_lst = [], []
        for p in ensemble_params_lst:
            agg_weight_lst.append(p["agg_weight"])
            w_local_lst.append(p["params"])
        return agg_weight_lst, w_local_lst

    def server_post_processing(self, w_glob, round_):
        """服务端后处理函数

        Args:
            w_glob :            dict
                                聚合后的全局参数

            round_ :            int or float
                                第x轮

        Returns:
            dict : Dict {
                'glob_params' : str
                                编码后的全局模型

                'code' :        int
                                状态码,

                'msg' :         str
                                状态消息,
            }
        """
        # encryption
        w_glob_b = pickle.dumps(w_glob)
        return self.encrypt.encode(w_glob_b)

    @staticmethod
    def cdw_feature_distance(old_model, new_model, device, train_loader):
        """cosine distance weight (cdw): calculate feature distance of
        the features of a batch of data by cosine distance.
        old_classifier,
        """
        old_model = old_model.to(device)
        # old_classifier = old_classifier.to(device)

        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            with torch.no_grad():
                # old_out = old_classifier(old_model(inputs))
                old_out = old_model(inputs)
                new_out = new_model(inputs)

            distance = 1 - torch.cosine_similarity(old_out, new_out)
            return torch.mean(distance).cpu().numpy()

    def server_exception(self, e):
        """服务端异常处理函数

        Args:
            e :                 Exception
                                异常消息

        Returns:
            dict : Dict {
                'glob_params' : str
                                编码后的全局模型

                'code' :        int
                                状态码,

                'msg' :         str
                                状态消息,
            }
        """
        print(e)
        print("检查客户端模型参数是否正常")
        return ""

    def server_ensemble(self, agg_weight_lst, w_local_lst, key_lst=None):
        """服务端集成函数

        Args:
            w_local_lst :       list
                                模型参数组成的list，model.state_dict()

            agg_weight_lst :    list
                                模型参数所占权重组成的list（该客户端聚合所占权重）

            key_lst :           list
                                需要聚合的网络层，fc.weight, fc.bias

        Returns:
            dict : w_glob
                   集成后的模型参数
        """
        if key_lst == None:
            all_local_key_lst = [set(w_local.keys()) for w_local in w_local_lst]
            key_lst = reduce(lambda x, y: x & y, all_local_key_lst)
        # sum up weights
        w_glob = {k: agg_weight_lst[0] * w_local_lst[0][k] for k in key_lst}
        for agg_weight, w_local in zip(agg_weight_lst[1:], w_local_lst[1:]):
            for k in key_lst:
                w_glob[k] += agg_weight * w_local[k]
        molecular = np.sum(agg_weight_lst)
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], molecular)
        return w_glob

    @abstractmethod
    def client(self, model_trainer, i, agg_weight=1.0):
        """获取客户端需要上传的模型参数及所占全局模型的权重

        Args:
            w_local :       collections.OrderedDict
                            模型参数，model.state_dict()

            agg_weight :    float
                            模型参数所占权重（该客户端聚合所占权重）

        Returns:
            dict : Dict {
                'params' :      collections.OrderedDict
                                上传的模型参数，model.state_dict()

                'agg_weight' :  float
                                模型参数所占权重（该客户端聚合所占权重）
            }
        """
        w_shared = {"params": {}, "agg_weight": agg_weight}
        # for k in self.shared_key_layers:
        #     w_shared['params'][k] = w_local[k].cpu()
        # return w_shared

        return NotImplemented

    @abstractmethod
    def server(self, ensemble_params_lst, round_):
        """服务端聚合客户端模型

        Args:
            ensemble_params_lst :   list
                                    每个客户端发送的参数组成的列表

            round_ :                int or float
                                    第x轮

        Returns:
            dict : Dict {
                'glob_params' : str
                                编码后的全局模型

                'code' :        int
                                状态码,

                'msg' :         str
                                状态消息,
            }
        """
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        # N, idxs_users, w_glob = self.server_pre_processing(w_local_lst)
        try:
            return NotImplemented
        except Exception as e:
            return self.server_exception(e)
        return self.server_post_processing(w_glob, round_)

    @abstractmethod
    def client_revice(self, model_trainer, w_glob_b):
        """客户端更新全局模型参数.

        Args:
            w_local :       collections.OrderedDict
                            模型参数，model.state_dict()

            w_glob_b :      bytes
                            服务器传来的二进制模型参数

        Returns:
            collections.OrderedDict
            更新后的模型参数，model.state_dict()
        """
        w_glob = pickle.loads(w_glob_b)
        # for k in self.shared_key_layers:
        #     w_local[k] = w_glob[k]
        return NotImplemented
