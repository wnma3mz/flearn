# coding: utf-8
# In Server
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np


def data2png(log_client_name, log_server_name):
    with open(log_client_name, "rb") as f:
        client_log_lst = pickle.load(f)
    with open(log_server_name, "r", encoding="utf-8") as f:
        server_log_lst = f.readlines()
    server_log_lst = [line.split(";")[1:] for line in server_log_lst]

    str2float = lambda item: float(item.split(":")[1].strip())
    server_acc_lst, server_loss_lst = [], []
    for line in server_log_lst:
        server_loss_lst.append(str2float(line[0]))
        server_acc_lst.append(str2float(line[1]))

    acc_lst = np.array([line[1] for line in client_log_lst]).T
    loss_lst = np.array([line[0] for line in client_log_lst]).T

    plt.cla()
    x_lst = range(1, len(server_acc_lst) + 1)
    plt.plot(x_lst, server_acc_lst, label="server mean", linestyle="--")
    for i, client_line in enumerate(acc_lst, 1):
        x_lst = range(1, len(client_line) + 1)
        plt.plot(x_lst, client_line, label="client {}".format(i))

    plt.xlabel("Round")
    plt.ylabel("Acc")
    plt.legend()
    plt.title("Test Acc")
    plt.savefig(log_client_name.split(".pkl")[0] + "-acc.png", dpi=300)

    plt.cla()
    x_lst = range(1, len(server_loss_lst) + 1)
    plt.plot(x_lst, server_loss_lst, label="server mean", linestyle="--")
    for i, client_line in enumerate(loss_lst, 1):
        x_lst = range(1, len(client_line) + 1)
        plt.plot(x_lst, client_line, label="client {}".format(i))

    plt.xlabel("Round")
    plt.ylabel("Loos")
    plt.legend()
    plt.title("Train Loss")
    plt.savefig(log_client_name.split(".pkl")[0] + "-loss.png", dpi=300)


if __name__ == "__main__":
    Round = 3
    N_clients = 10
    iid = True
    batch_size = 128
    dataset_name = "covid"
    log_name = "log_round{}_clients{}_iid{}_dataset-{}"
    log_server_name = log_name.format(Round, N_clients, iid, dataset_name)
    log_client_name = log_server_name + ".pkl"

    data2png(log_client_name, log_server_name)
