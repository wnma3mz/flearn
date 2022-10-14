# coding: utf-8
import copy

from flearn.client import Client


class LGClient(Client):
    def receive(self, i, glob_params):
        self.w_local_bak = copy.deepcopy(self.trainer.model.state_dict())
        return super(LGClient, self).receive(i, glob_params)

    def evaluate(self, i):
        # 跳过测试
        # if i < 100:
        #     return {
        #         "code": 200,
        #         "msg": "Model evaluate completed",
        #         "client_id": self.client_id,
        #         "test_acc": [0, 0],
        #     }
        test_acc_lst = [self.trainer.test(loader)[1] for loader in self.testloader]

        if self.save:
            self.trainer.save(self.agg_fpath)

        if self.best_acc < test_acc_lst[0]:
            self.trainer.save(self.best_fpath)
            self.best_acc = test_acc_lst[0]
        else:
            # referring to https://github.com/pliang279/LG-FedAvg/blob/7af0568b2cae88922ebeacc021b1679815092f4e/main_lg.py#L139
            # 不更新模型
            self.trainer.model.load_state_dict(self.w_local_bak)

        if self.log == True and "train_loss" in self.__dict__.keys():
            test_acc_str = "; ".join("{:.4f}".format(x) for x in test_acc_lst)
            log_i = i, self.train_loss, self.train_acc, test_acc_str
            self.log_client.logger.info(self.log_fmt.format(*log_i))
            del self.__dict__["train_loss"]
        return {
            "code": 200,
            "msg": "Model evaluate completed",
            "client_id": self.client_id,
            "test_acc": test_acc_lst,
        }
