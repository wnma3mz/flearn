# coding: utf-8
import copy

from flearn.client import Client


class DistillClient(Client):
    def revice(self, i, glob_params):
        # decode
        data_glob_d = self.strategy.revice_processing(glob_params)
        # update
        update_w, logits_glob = self.strategy.client_revice(self.trainer, data_glob_d)

        if self.scheduler != None:
            self.scheduler.step()
        self.trainer.model.load_state_dict(update_w)
        self.trainer.glob_logit = copy.deepcopy(logits_glob)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
