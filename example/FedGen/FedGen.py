# coding: utf-8
import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from flearn.client import Client
from flearn.common import Trainer
from flearn.common.strategy import AVG


class Gen(AVG):
    """
    Federated Distillation via Generative Learning

    [1] Zhu Z, Hong J, Zhou J. Data-Free Knowledge Distillation for Heterogeneous Federated Learning[J]. arXiv preprint arXiv:2105.10056, 2021.
    """

    def __init__(self, model_fpath, model_base, generative_model, optimizer, device):
        super(Gen, self).__init__(model_fpath)
        self.model_base = model_base
        self.generative_model = generative_model
        self.optimizer = optimizer
        self.device = device

        self.model_base.to(self.device)
        self.model_base.eval()

        self.generative_model.to(self.device)

    def client(self, trainer, agg_weight=1.0):
        w_shared = super(Gen, self).client(trainer, agg_weight)
        # 上传客户端的标签数量
        w_shared["label_counts"] = trainer.label_counts
        return w_shared

    @staticmethod
    def visualize_images(generator, glob_iter, available_labels, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f"images/iter{glob_iter}.png"
        y = np.repeat(available_labels, repeats=repeats, axis=0)
        y_input = torch.tensor(y)
        generator.eval()
        with torch.no_grad():
            images = generator(y_input)["output"]  # 0,1,..,K, 0,1,...,K
        images = images.view(repeats, -1, *images.shape[1:])
        images = images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        # print("Image saved to {}".format(path))

    @staticmethod
    def train_gan(
        generative_model, optimizer, student_model, discriminator_model_lst, **kwargs
    ):
        """训练GAN
        generative_model: 生成模型
        optimizer: 生成模型的优化器
        student_model: 全局聚合后的模型
        discriminator_model_lst: 每个客户端的模型
        """
        qualified_labels = kwargs["qualified_labels"]
        label_weights = kwargs["label_weights"]
        batch_size = kwargs["batch_size"]
        unique_labels = kwargs["unique_labels"]
        alpha, beta, eta = kwargs["alpha"], kwargs["beta"], kwargs["eta"]
        device = kwargs["device"]
        start_layer_idx = kwargs["start_layer_idx"]

        iter_ = 5
        epochs = 50 // iter_
        generative_model.train()

        NLL_loss = nn.NLLLoss(reduce=False)
        CE_loss = nn.CrossEntropyLoss()

        def get_teacher_loss(y_input, gen_output):
            teacher_loss = 0
            teacher_logit = 0
            for user_idx, discriminator_model in enumerate(discriminator_model_lst):
                discriminator_model.eval()
                weight = label_weights[y][:, user_idx].reshape(-1, 1)
                expand_weight = np.tile(weight, (1, len(unique_labels)))
                with torch.no_grad():
                    dis_output = discriminator_model(gen_output, start_layer_idx)
                user_output_logp_ = F.log_softmax(dis_output, dim=1)
                teacher_loss_ = torch.mean(
                    NLL_loss(user_output_logp_, y_input)
                    * torch.tensor(weight, dtype=torch.float32)
                )
                teacher_loss += teacher_loss_
                teacher_logit += dis_output * torch.tensor(
                    expand_weight, dtype=torch.float32
                )
            return teacher_loss, teacher_logit

        for _ in range(epochs):
            for _ in range(iter_):
                y = np.random.choice(qualified_labels, batch_size)
                y_input = torch.tensor(y).to(device)
                ## feed to generator

                gen_result = generative_model(y_input, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps = gen_result["output"], gen_result["eps"]
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                # encourage different outputs
                diversity_loss = generative_model.diversity_loss(eps, gen_output)

                ######### get teacher loss ############
                teacher_loss, teacher_logit = get_teacher_loss(y_input, gen_output)

                ######### get student loss ############

                if beta > 0:
                    with torch.no_grad():
                        student_output = student_model(gen_output, start_layer_idx)
                    student_output = student_output.cpu().detach()
                    student_loss = F.kl_div(
                        F.log_softmax(student_output, dim=1),
                        F.softmax(teacher_logit, dim=1),
                    )

                    loss = (
                        alpha * teacher_loss
                        - beta * student_loss
                        + eta * diversity_loss
                    )
                else:
                    loss = alpha * teacher_loss + eta * diversity_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return generative_model

    @staticmethod
    def get_label_weights(label_counts_lst, unique_labels):
        MIN_SAMPLES_PER_LABEL = 1
        label_weights = []
        qualified_labels = []
        for label in unique_labels:
            weights = [label_counts[label] for label_counts in label_counts_lst]
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((len(unique_labels), -1))
        return label_weights, qualified_labels

    def server(self, ensemble_params_lst, round_, **kwargs):
        """
        kwargs: {
            "batch_size": batch_size,
            "alpha": 1,
            "beta": 0,
            "eta": 1,
            "device": device,
        }
        """
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        label_counts_lst = self.extract_lst(ensemble_params_lst, "label_counts")

        unique_labels = []
        for x in label_counts_lst:
            unique_labels += list(x.keys())
        unique_labels = list(set(unique_labels))
        label_weights, qualified_labels = self.get_label_weights(
            label_counts_lst, unique_labels
        )
        kwargs["qualified_labels"] = qualified_labels
        kwargs["label_weights"] = label_weights
        kwargs["unique_labels"] = unique_labels

        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)

        self.model_base.load_state_dict(w_glob)
        student_model = copy.deepcopy(self.model_base)

        discriminator_model_lst = []
        for w_local in w_local_lst:
            self.model_base.load_state_dict(w_local)
            discriminator_model_lst.append(copy.deepcopy(self.model_base))

        self.generative_model = self.train_gan(
            self.generative_model,
            self.optimizer,
            student_model,
            discriminator_model_lst,
            **kwargs,
        )

        self.visualize_images(self.generative_model, round_, unique_labels, repeats=10)

        return {"w_glob": w_glob, "gen_model": self.generative_model.state_dict()}

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_glob = data_glob_d["w_glob"]
        for k in w_glob.keys():
            w_local[k] = w_glob[k]

        w_gen_glob = data_glob_d["gen_model"]
        self.generative_model.load_state_dict(w_gen_glob)
        return w_local, self.generative_model


class GenClient(Client):
    def revice(self, i, glob_params):
        # decode
        data_glob_d = self.strategy.revice_processing(glob_params)

        # update
        update_w, update_gen = self.strategy.client_revice(self.trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        self.trainer.model.load_state_dict(update_w)

        self.trainer.generative_model = update_gen
        self.trainer.generative_model.to(self.device)
        self.trainer.generative_model.eval()

        generative_alpha = self.trainer.generative_alpha
        generative_beta = self.trainer.generative_beta

        self.trainer.generative_alpha = self.exp_lr_scheduler(
            i, decay=0.98, init_lr=generative_alpha
        )
        self.trainer.generative_beta = self.exp_lr_scheduler(
            i, decay=0.98, init_lr=generative_beta
        )

        if self.save:
            self.trainer.save(self.agg_fpath)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr


class GenTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(GenTrainer, self).__init__(
            model, optimizer, criterion, device, display=display
        )
        self._init_label_counts()
        self._init_label_counts_flag = True
        self.early_stop = 100  # GAN提前结束计算loss
        self.generative_model = None
        self.generative_alpha = 1  # 初始值
        self.generative_beta = 1  # 初始值

        self.start_layer_idx = 10  # cifar10
        # self.start_layer_idx = 8  # mnist

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.gen_loss = nn.NLLLoss(reduce=False)  # same as above
        # self.gen_loss = self.criterion  # same as above

    def _init_label_counts(self):
        # cifar10
        self.available_labels = list(range(10))
        self.label_counts = {}
        for i in self.available_labels:
            # 确保每个都有一定的weight
            self.label_counts[i] = 1

    def batch(self, data, target):
        output = self.model(data)
        loss = self.criterion(output, target)

        if self.model.training:
            if self.generative_model != None:
                ### get generator output(latent representation) of the same label
                with torch.no_grad():
                    gen_output = self.generative_model(target)["output"]
                dis_output = self.model(gen_output, self.start_layer_idx)
                target_p = F.softmax(dis_output, dim=1).clone().detach()
                user_latent_loss = self.generative_beta * self.kl_loss(
                    F.log_softmax(output, dim=1), target_p
                )

                # self.gen_batch_size = len(target)
                sampled_y = np.random.choice(self.available_labels, len(target))
                sampled_y = torch.tensor(sampled_y).to(self.device)
                # latent representation when latent = True, x otherwise

                with torch.no_grad():
                    gen_output = self.generative_model(sampled_y)["output"]
                dis_output = self.model(gen_output, self.start_layer_idx)
                teacher_loss = self.generative_alpha * torch.mean(
                    self.gen_loss(F.log_softmax(dis_output, dim=1), sampled_y)
                )
                # this is to further balance oversampled down-sampled synthetic data
                # gen_ratio = self.gen_batch_size / len(target)
                gen_ratio = 1
                print(
                    loss.data.item(),
                    teacher_loss.data.item(),
                    user_latent_loss.data.item(),
                )
                loss += gen_ratio * teacher_loss + user_latent_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc

    def train(self, data_loader, epochs=1):
        # 假设每次的trainloader相同，所以仅需进行一次统计
        if self._init_label_counts_flag:
            for _, y in data_loader:
                unique_y, counts = torch.unique(y, return_counts=True)
                labels = unique_y.detach().numpy()
                counts = counts.detach().numpy()
                for label, count in zip(labels, counts):
                    self.label_counts[int(label)] += count
            self._init_label_counts_flag = False
        return super(GenTrainer, self).train(data_loader, epochs=epochs)
