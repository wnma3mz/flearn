# coding: utf-8
import copy

from flearn.client import Client


class LSDClient(Client):
    def receive(self, i, glob_params):
        # decode
        data_glob_d = self.strategy.receive_processing(glob_params)

        # update
        update_w = self.strategy.client_receive(self.trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        self.trainer.model.load_state_dict(update_w)

        self.trainer.teacher_model = copy.deepcopy(self.trainer.model)
        self.trainer.teacher_model.load_state_dict(update_w)
        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
