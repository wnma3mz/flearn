# coding: utf-8
import copy

from flearn.common.distiller import MDDistiller
from flearn.common.strategy import LG_R

from .utils import convert_to_tensor


class MD(LG_R):
    def __init__(self, shared_key_layers, glob_model=None, device=None):
        super(MD, self).__init__(shared_key_layers)
        self.glob_model = glob_model
        if self.glob_model:
            self.device = device
            self.glob_model.to(device)
            self.glob_model.train()
        # else:
        #     print("Warning: glob model is None")

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_local_bak = copy.deepcopy(w_local)

        distiller = MDDistiller(self.glob_model, w_local, self.device)
        w_glob_model = distiller.run(data_glob_d, epoch=1)

        for k in w_glob_model.keys():
            w_local_bak[k] = w_glob_model[k]

        return w_local_bak

    def client_pub_predict(self, w_local_lst, **kwargs):
        w_local_lst = [convert_to_tensor(w_local) for w_local in w_local_lst]
        distiller = MDDistiller(self.glob_model, None, self.device)
        return distiller.predict(w_local_lst, kwargs["data_loader"])

    def server(self, ensemble_params_lst, round_, **kwargs):
        w_local_lst = self.extract_lst(ensemble_params_lst, "params")
        x_lst, logits_lst = self.client_pub_predict(w_local_lst, **kwargs)
        return {"x_lst": x_lst, "logits_lst": logits_lst, "w_glob": ""}
