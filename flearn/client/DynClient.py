# coding: utf-8
import copy

from flearn.client import Client


class DynClient(Client):
    def revice(self, i, glob_params):
        w_local = self.trainer.weight
        self.w_local_bak = copy.deepcopy(w_local)

        data_glob_d = self.strategy.revice_processing(glob_params)
        # update
        update_w = self.strategy.client_revice(self.trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        # self.trainer.model.load_state_dict(self.w_local_bak)
        self.trainer.model.load_state_dict(update_w)
        self.trainer.server_state_dict = copy.deepcopy(update_w)
        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
