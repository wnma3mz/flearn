# coding: utf-8
import copy

import numpy as np

from flearn.common.distiller import PAVDistiller

from .lg_reverse import LG_R
from .utils import cdw_feature_distance, convert_to_np, convert_to_tensor


class PAV(LG_R):
    """
    FedPAV, https://github.com/cap-ntu/FedReID

    note:
    1. soft label or cosine distance
    2. Knowledge Distiller

    References
    ----------
    .. [1] Zhuang W, Wen Y, Zhang X, et al. Performance Optimization of Federated Person Re-identification via Benchmark Analysis[C]//Proceedings of the 28th ACM International Conference on Multimedia. 2020: 955-963.
    """

    def client(self, trainer, old_model, device, trainloader):
        """客户端发送参数

        Args:
            trainer :      Trainer
                                 客户端的训练器

            old_model :          Model
                                 客户端训练前的模型（上一轮更新后的模型）

            device :
                                 训练使用GPU or CPU

            trainloader :        数据集
                                 本地的训练集，仅使用一轮

        Returns:
            dict : Dict {
                'params' :      collections.OrderedDict
                                上传的模型参数，model.state_dict()

                'agg_weight' :  float
                                模型参数所占权重（该客户端聚合所占权重）
            }
        """
        distance = cdw_feature_distance(old_model, trainer.model, device, trainloader)
        # 权重值太小，x100处理
        w_shared = {"agg_weight": np.float(distance) * 100}
        w_local = convert_to_np(trainer.weight)
        w_shared["params"] = {k: v for k, v in w_local.items() if k not in self.shared_key_layers}
        return w_shared

    @staticmethod
    def load_model(glob_model, w_dict):
        glob_model_dict = glob_model.state_dict()
        glob_model_dict.update(w_dict)
        glob_model.load_state_dict(glob_model_dict)
        return glob_model

    def pav_kd(self, w_local_lst, w_glob, **kwargs):
        # 进行蒸馏
        if "kd" in kwargs.keys() and kwargs["kd"] == True:
            self.distiller = PAVDistiller(
                kwargs["input_len"],
                kwargs["kd_loader"],
                kwargs["device"],
                kwargs["regularization"],
            )

            # 全局模型不完整，所以手动拼接Sequential
            glob_model = kwargs["glob_model"]

            client_lst = []
            # 客户端参数转模型
            for w_local in w_local_lst:
                glob_model = self.load_model(glob_model, convert_to_tensor(w_local))
                client_lst.append(copy.deepcopy(glob_model))

            glob_model = self.load_model(glob_model, convert_to_tensor(w_glob))
            # 知识蒸馏+正则化
            glob_model = self.distiller.run(client_lst, glob_model)
            # 模型转回参数
            glob_model_d = glob_model.state_dict()
            for k in w_glob.keys():
                w_glob[k] = glob_model_d[k].cpu()

        return w_glob

    def server_post_processing(self, ensemble_params_lst, ensemble_params, **kwargs):
        w_local_lst = self.extract_lst(ensemble_params_lst, "params")
        ensemble_params["w_glob"] = self.pav_kd(w_local_lst, ensemble_params["w_glob"], **kwargs)

        return ensemble_params

    def server(self, ensemble_params_lst, round_, **kwargs):
        """服务端聚合客户端模型并蒸馏
        Args:
            kwargs :            dict
                                蒸馏所需参数
        """
        ensemble_params = super().server(ensemble_params_lst, round_)
        return self.server_post_processing(ensemble_params_lst, ensemble_params, **kwargs)
