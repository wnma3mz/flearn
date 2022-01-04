# coding: utf-8
import concurrent.futures

import numpy as np
import torch

from flearn.client import Client
from flearn.common import Trainer
from flearn.common.strategy import Strategy
from flearn.server import Communicator as sc
from flearn.server import Server


class SLServer(Server):
    def forward_backward(self, item, round_, k=-1, **kwargs):
        # for item in data_lst:
        if int(round_) != int(item["round"]):
            return ""
        model_data = self.strategy.revice_processing(item["datas"])

        self.strategy.model.train()
        glob_params = self.strategy.server(model_data, round_, **kwargs)
        glob_params["client_id"] = item["client_id"]
        return self.strategy.upload_processing(glob_params)

    def test(self):
        pass


class SLC(sc):
    def get_data_lst(self, command, json_d):
        data_lst = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_url = {
                executor.submit(self.thread_func, url, command, json_d[idx]): url
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
        json_d = [{"round": ri}] * len(self.client_url_lst)
        # 发送训练指令
        data_lst = self.get_data_lst("train", json_d)

        # 发送上传指令
        data_lst = self.get_data_lst("upload", json_d)

        # 聚合
        train_acc_lst, loss_lst = []
        for item in data_lst:
            glob_params = self.server.forward_backward(item, ri, k=k, **kwargs)
            if json_d["grads"] == "":
                raise SystemError("Aggregation error!")
            json_d[glob_params["client_id"]]["grads"] = glob_params
            loss_lst.append(glob_params["loss"])
            train_acc_lst.append(glob_params["acc"])

        # 发送参数，客户端接收
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
        loss, train_acc = self.server.mean_lst(loss_lst), self.server.mean_lst(
            train_acc_lst
        )
        log_msg = ri, loss, train_acc, test_acc
        if self.log:
            self.log_server.logger.info(self.log_fmt.format(*log_msg))
        return loss, train_acc, test_acc


class SL(Strategy):
    def __init__(
        self, model_fpath, model_server, criterion_server, optim_server, device
    ):
        super(SL, self).__init__(model_fpath)
        self.device = device
        self.model_server = model_server
        self.criterion_server = criterion_server
        self.optim_server = optim_server
        self.model_server.to(self.device)

    def client(self, output, target, agg_weight=1):
        client_output = output.clone().detach().requires_grad_(True)
        return {
            "output_grads": client_output,
            "target": target,
            "is_train": True,
        }

    def client_revice(self, data_glob_d):
        grads = data_glob_d["grads"]
        return grads

    def server(self, params, round_):

        target = params["target"].to(self.device)
        client_output = params["output_grads"].to(self.device)

        output = self.model_server(client_output)
        loss = self.criterion_server(output, target)

        if params["is_train"]:
            self.optim_server.zero_grad()
            loss.backward()
            self.optim_server.step()
            client_grad = client_output.grad.clone().detach()
            return {
                "grads": client_grad,
                "acc": (output.data.max(1)[1] == target.data).sum().item(),
                "loss": loss.data.item(),
            }
        return {
            "acc": (output.data.max(1)[1] == target.data).sum().item(),
            "loss": loss.data.item(),
        }


class SLTrainer(Trainer):
    def _iteration(self, loader):
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            yield self.model(data), target.cpu()

    def train(self, data_loader):
        self.model.train()
        self.is_train = True
        with torch.enable_grad():
            return self._iteration(data_loader)

    def backward(self, client_output_tmp, client_grad):
        self.optimizer_.zero_grad()
        client_output_tmp.backward(client_grad)
        self.optimizer_.step()


class SLClient(Client):
    def train(self, i):
        output, target = self.trainer.train(self.trainloader)
        self.upload_model = self.strategy.client(output, target, agg_weight=1.0)
        self.output = output  # 反向传播
        return self._pickle_model()

    def _pickle_model(self):
        if self.save:
            self.trainer.save(self.update_fpath)

        return {
            "code": 200,
            "msg": "Model complete the training",
            "client_id": self.client_id,
        }

    def revice(self, i, glob_params):
        # decode
        data_glob_d = self.strategy.revice_processing(glob_params)

        # update
        grads = self.strategy.client_revice(self.trainer, data_glob_d)
        self.trainer.backward(self.output, grads)

        if self.save:
            self.trainer.save(self.agg_fpath)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
