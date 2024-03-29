import copy

from flearn.client import Client


class MOONClient(Client):
    def __init__(self, conf, pre_buffer_size=1):
        super(MOONClient, self).__init__(conf)
        # 保存以前模型的大小
        self.pre_buffer_size = pre_buffer_size
        # 记录运行轮数
        self.ci = -1

    def train(self, i):
        # 每轮训练+1
        self.ci += 1
        self.train_loss, self.train_acc = self.trainer.train(self.trainloader, self.epoch)
        # 权重为本地数据大小
        self.upload_model = self.strategy.client(self.trainer, agg_weight=len(self.trainloader))
        return self._pickle_model()

    def receive(self, i, glob_params):
        # 额外需要两类模型，glob和previous，一般情况下glob只有一个，previous也定义只有一个
        # 如果存储超过这个大小，则删除最老的模型
        while len(self.trainer.previous_model_lst) >= self.pre_buffer_size:
            self.trainer.previous_model_lst.pop(0)
        self.trainer.previous_model_lst.append(copy.deepcopy(self.trainer.model))

        # decode
        data_glob_d = self.strategy.receive_processing(glob_params)

        # update
        update_w = self.strategy.client_receive(self.trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        self.trainer.model.load_state_dict(update_w)

        # 如果该客户端训练轮数不等于服务器端的训练轮数，则表示该客户端的模型本轮没有训练，则不做对比学习，并且同步进度轮数。
        if self.ci != i:
            self.trainer.global_model = None
            self.trainer.previous_model_lst = []
            self.ci = i - 1
        else:
            self.trainer.global_model = copy.deepcopy(self.trainer.model)
            self.trainer.global_model.load_state_dict(update_w)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
