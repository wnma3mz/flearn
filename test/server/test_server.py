# coding: utf-8
import base64
import json
import pickle
import sys

sys.path.append("../../..")
import torch
from flearn.common import Encrypt
from flearn.server import Server
from flearn.server.communicate import Communicate as sc

if __name__ == "__main__":
    conf_fpath = "server_settings.json"
    with open(conf_fpath, "r", encoding="utf-8") as f:
        conf = json.loads(f.read())
    s_model = Server(conf)
    params = {
        "params": {
            "fc.weight": torch.tensor([1, 2, 3]),
            "fc.bias": torch.tensor([2, 4, 5]),
        },
        "agg_weight": 1.0,
    }
    model_parambs_b64 = base64.b64encode(pickle.dumps(params))
    model_b64_str = model_parambs_b64.decode()
    data_lst = [
        {
            "datas": model_b64_str,
            "round": str(0),
            "client_id": "foo",
        }
    ]
    glob_m = s_model.ensemble(data_lst, 0)

    server_o = sc(conf_fpath)
