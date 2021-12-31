# coding: utf-8
import os
import pickle
import re
from os.path import join as ospj

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.pyplot import MultipleLocator

# mpl.use("Agg")

font = {"size": 22}
rc("font", **font)
cmaps = (
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "Pastel1",
    "Pastel2",
    "Paired",
    "Set1",
    "Set2",
    "Set3",
    "Accent",
    "Dark2",
)
cmap = cmaps[-4]


def read_data(fname):
    with open(fname, "r") as f:
        client_log_lst = f.readlines()

    client_acc_lst, client_loss_lst = [], []
    for line in client_log_lst:
        res = loss_re.findall(line)
        if res:
            client_loss_lst.append(float(res[0]))
        res = acc_re.findall(line)
        if res:
            client_acc_lst.append(float(res[0]))

    x_lst = range(1, len(client_acc_lst) + 1)
    return x_lst, client_loss_lst, client_acc_lst


def plot_line_acc():

    name = "Amazon"
    suffix = ""

    plt.cla()
    color_i = 0

    log_name = "log_strategy_avg_client{}_dataset_Caltech-10{}"
    x_lst, client_loss_lst, client_acc_lst = read_data(
        ospj(root_dir, log_name.format(name, suffix))
    )
    plt.plot(
        x_lst,
        client_acc_lst,
        label="FedAVG",
        linestyle="-",
        color=plt.get_cmap(cmap).colors[color_i],
    )
    color_i += 1

    log_name = "log_strategy_lg_r_client{}_dataset_Caltech-10{}"
    x_lst, client_loss_lst, client_acc_lst = read_data(
        ospj(root_dir, log_name.format(name, suffix))
    )
    plt.plot(
        x_lst,
        client_acc_lst,
        label="FedLG(共享特征层)",
        linestyle="-",
        color=plt.get_cmap(cmap).colors[color_i],
    )
    color_i += 1

    log_name = "log_strategy_lg_client{}_dataset_Caltech-10{}"
    x_lst, client_loss_lst, client_acc_lst = read_data(
        ospj(root_dir, log_name.format(name, suffix))
    )
    plt.plot(
        x_lst,
        client_acc_lst,
        label="FedLG(共享全连接层)",
        linestyle="-",
        color=plt.get_cmap(cmap).colors[color_i],
    )
    color_i += 1

    plt.xlabel("Round")
    plt.ylabel("测试集性能Acc")
    plt.legend()
    plt.title("Caltech-10任务中Amazon客户端不同方法的收敛速度对比")
    plt.show()


def plot_bar():

    plt.cla()

    xi = 0
    bar_width = 0.3
    name_list = ["Amazon", "Caltech", "DSLR", "Webcam"]

    suffix = ""
    avg_lst, lg_lst, lg_r_lst = [], [], []
    for name in name_list:
        log_name = "log_strategy_avg_client{}_dataset_Caltech-10{}"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name, suffix))
        )
        avg_lst.append(max(client_acc_lst))

        log_name = "log_strategy_lg_r_client{}_dataset_Caltech-10{}"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name, suffix))
        )
        lg_r_lst.append(max(client_acc_lst))

        log_name = "log_strategy_lg_client{}_dataset_Caltech-10{}"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name, suffix))
        )
        lg_lst.append(max(client_acc_lst))

    x = list(range(len(avg_lst)))
    color_i = 0
    total_width, n = 0.6, 3
    width = total_width / n

    x, color_i = single_dataset_bar(x, avg_lst, "FedAVG", color_i)
    x, color_i = single_dataset_bar(x, lg_lst, "FedLG(共享全连接层)", color_i)
    x, color_i = single_dataset_bar(x, lg_r_lst, "FedLG(共享特征层)", color_i)

    plt.legend(loc="best", fontsize=12)
    plt.ylabel("测试集性能Acc")
    plt.title("Caltech-10任务中不同方法的精度对比")
    plt.show()


def single_dataset_bar(x, data_lst, label, color_i):
    plt.bar(
        x,
        data_lst,
        width=width,
        label=label,
        color=plt.get_cmap(cmap).colors[color_i],
        tick_label=name_list,
    )
    for a, b in zip(x, data_lst):  # 柱子上的数字显示
        plt.text(a, b, "%.3f" % b, ha="center", va="bottom", fontsize=10)
    for i in range(len(x)):
        x[i] = x[i] + width
    color_i += 1
    return x, color_i


def plot_bar_more():

    plt.cla()

    xi = 0
    bar_width = 0.3
    name_list = ["Amazon", "Caltech", "DSLR", "Webcam"]

    avg_lst, lg_r_lst, kd_r_t2s_lst, kd_r_t2t_lst = [], [], [], []
    for name in name_list:
        log_name = "log_strategy_avg_client{}_dataset_Caltech-10"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name))
        )
        avg_lst.append(max(client_acc_lst))

        log_name = "log_strategy_lg_r_client{}_dataset_Caltech-10"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name))
        )
        lg_r_lst.append(max(client_acc_lst))

        log_name = "log_strategy_kd_r_client{}_dataset_Caltech-10epoch3"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name))
        )
        kd_r_t2s_lst.append(max(client_acc_lst))

        suffix = "epoch3_teacher_kd_self"
        log_name = "log_strategy_kd_r_client{}_dataset_Caltech-10{}"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name, suffix))
        )
        kd_r_t2t_lst.append(max(client_acc_lst))

    x = list(range(len(avg_lst)))
    color_i = 0
    total_width, n = 0.8, 4
    width = total_width / n

    x, color_i = single_dataset_bar(x, avg_lst, "FedAVG", color_i)
    x, color_i = single_dataset_bar(x, lg_r_lst, "FedLG", color_i)
    x, color_i = single_dataset_bar(x, kd_r_t2s_lst, "FedLKD(服务器端蒸馏本地模型)", color_i)
    x, color_i = single_dataset_bar(x, kd_r_t2t_lst, "FedLKD(服务器端蒸馏服务器端)", color_i)

    plt.legend(loc="best", fontsize=10)
    plt.ylabel("测试集性能Acc")
    plt.title("Caltech-10任务中不同方法的精度对比")
    plt.show()


if __name__ == "__main__":
    Round = 3
    client_numbers = 10
    iid = True
    batch_size = 128
    dataset_name = "covid"

    root_dir = r"C:\Users\lu\Desktop\logdata"
    acc_re = re.compile(r"TestAcc: (.+?);")
    loss_re = re.compile(r"Loss: (.+?);")

    name_list = ["Amazon", "Caltech", "DSLR", "Webcam"]

    suffix = "epoch3_bn_alexnet_teacher_kd_self"
    avg_lst, lg_r_lst, kd_r_t2s_lst, kd_r_t2t_lst = [], [], [], []
    for name in name_list:
        log_name = "log_strategy_kd_r_client{}_dataset_Caltech-10{}"
        x_lst, client_loss_lst, client_acc_lst = read_data(
            ospj(root_dir, log_name.format(name, suffix))
        )
        print(name, max(client_acc_lst))
