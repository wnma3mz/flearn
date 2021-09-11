# coding: utf-8
import glob
import os
import numpy as np


if __name__ == "__main__":
    dir_ = "D:\shared_files\share_learning\contrast_experiments\BN\DomainNet"
    flst = glob.glob(os.path.join(dir_, "log*"))
    for log_server_name in flst:
        with open(log_server_name, "r", encoding="utf-8") as f:
            server_log_lst = f.readlines()

        server_log_lst = [line.split(";")[1:] for line in server_log_lst]

        str2float = lambda item: float(item.split(":")[1].strip())
        server_acc_lst = [str2float(line[3]) for line in server_log_lst]

        name = os.path.basename(log_server_name).split("client")[1].split("_")[0]
        value = max(server_acc_lst)
        print("{:>10}: {:>1.2f}".format(name, value))
