# coding: utf-8
import copy

from flearn.client import Client


class ProxClient(Client):
    """"FedProx, 结合ProxTrainer使用"""

    def revice(self, i, glob_params):
        w_local = self.model_trainer.weight
        self.w_local_bak = copy.deepcopy(w_local)
        # decode
        data_glob_d = self.strategy.revice_processing(glob_params)
        # update
        update_w = self.strategy.client_revice(self.model_trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        # self.model_trainer.model.load_state_dict(self.w_local_bak)
        self.model_trainer.model.load_state_dict(update_w)
        self.model_trainer.server_model = copy.deepcopy(self.model_trainer.model)
        self.model_trainer.server_model.load_state_dict(update_w)

        self.model_trainer.server_model.eval()
        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
