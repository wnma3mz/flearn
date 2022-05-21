# coding: utf-8
import copy

import numpy as np

from flearn.common.distiller import DFDistiller, KDLoss

from .strategy import ParentStrategy
from .utils import convert_to_tensor


class DF(ParentStrategy):
    """
    Ensemble distillation for robust model fusion in federated learning

    [1] Lin T, Kong L, Stich S U, et al. Ensemble distillation for robust model fusion in federated learning[J]. arXiv preprint arXiv:2006.07242, 2020.
    """

    def __init__(self, model_base, strategy):
        super().__init__(strategy)
        self.model_base = model_base

    def server_post_processing(self, ensemble_params_lst, ensemble_params, **kwargs):
        w_glob = convert_to_tensor(ensemble_params["w_glob"])
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)

        teacher_lst = []
        for w_local in w_local_lst:
            self.model_base.load_state_dict(convert_to_tensor(w_local))
            teacher_lst.append(copy.deepcopy(self.model_base))

        self.model_base.load_state_dict(w_glob)
        student = copy.deepcopy(self.model_base)

        kd_loader, device = kwargs.pop("kd_loader"), kwargs.pop("device")
        temperature = kwargs.pop("T")
        distiller = DFDistiller(
            kd_loader,
            device,
            kd_loss=KDLoss(temperature),
        )

        molecular = np.sum(agg_weight_lst)
        weight_lst = [w / molecular for w in agg_weight_lst]
        # agg_weight_lst：应该依照每个模型在验证集上的性能来进行分配
        ensemble_params["w_glob"] = distiller.multi(
            teacher_lst, student, kwargs.pop("method"), weight_lst=weight_lst, **kwargs
        )
        return ensemble_params

    def server(self, ensemble_params_lst, round_, **kwargs):
        """
        kwargs:       dict
                        {
                            "lr":    学习率,
                            "T":     蒸馏超参，温度
                            "epoch": 蒸馏训练轮数
                            "method": 多个教师蒸馏一个学习的方法，avg_logits, avg_losses
                            "kd_loader": 蒸馏数据集，仅需输入，无需标签
                        }
        """
        ensemble_params = super().server(ensemble_params_lst, round_)
        return self.server_post_processing(
            ensemble_params_lst, ensemble_params, **kwargs
        )
