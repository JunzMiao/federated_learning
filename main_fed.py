import copy
# from random import shuffle
# from site import USER_SITE
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
# from torch import nn
from torch.nn import init
import time


import data
import model
import train
import metric as M

import em_client

# NUM_USERS = 10
# FRAC = 0.3

MODEL = "CNN" # "MLP" "GRU" "CNN"
USE_FL = True # True False

# USER_RATIOS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
USER_RATIOS = [1, 1, 1]
# USER_RATIOS = [1 for _ in range(10)]
NUM_USERS = len(USER_RATIOS)
FRAC = 1

EPOCH = 10
LOCAL_EP = 5
LOCAL_BS = 32
TEST_BS = LOCAL_BS
DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

if MODEL == "MLP":
    NET = model.MLP(41, 48, 5).to(DEVICE)
elif MODEL == "GRU":
    NET = model.GRU_MLP(41, 5).to(DEVICE)
elif MODEL == "CNN":
    NET = model.CNN(41, 5).to(DEVICE)
else:
    assert False

# def idx_split_iid(total, num_users):
#     num_items = int(total/num_users)
#     dict_users, all_idxs = {}, [i for i in range(total)]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def idx_split(total, ratios):
    t = sum(ratios)
    num_items = list(map(lambda r : int(total * (r/t)), ratios))
    num_users = len(ratios)
    dict_users, all_idxs = {}, [i for i in range(total)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    dict_users[0] = dict_users[0].union(set(all_idxs))
    assert sum(map(lambda k : len(dict_users[k]), dict_users)) == total
    for i in range(num_users):
        dict_users[i] = list(map(int, dict_users[i]))
    return dict_users

def non_iid_ids_split(df):
    idxs = [[] for _ in range(5)]
    ratios = [0 for _ in range(5)]
    for i in range(len(df)):
        attack_map = df.iloc[i]["attack_map"]
        idxs[attack_map].append(i)
        ratios[attack_map] += 1
    
    normal_idxs = idxs[0]
    idxs = idxs[1:]
    attack_ratios = ratios[1:]
    # attack_ratios = list(map(lambda x : x/sum(attack_ratios), attack_ratios))
    dict_users = {}
    
    dic_normal = idx_split(len(normal_idxs), attack_ratios)
    for i in range(4):
        dict_users[i] = idxs[i]
        for j in dic_normal[i]:
            dict_users[i].append(normal_idxs[j])
    
    # n = sum(map(lambda x : ))
    # assert(sum(ratios) == len(df))
    return dict_users, attack_ratios

    

    # type2idx_map = {
    #     ""
    # }

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# training
def FedAvg(l_params, ratios):
    print(f"[*] {len(l_params)} - {len(ratios)}")
    assert(len(l_params) == len(ratios))
    avg_params = copy.deepcopy(l_params[0]) 
    for k in avg_params.keys():
        avg_params[k] = torch.mul(avg_params[k], ratios[0])
    for k in avg_params.keys():
        for i in range(1, len(l_params)):
            avg_params[k] += torch.mul(l_params[i][k], ratios[i])
        avg_params[k] = torch.div(avg_params[k], sum(ratios))
    return avg_params


# train_df = pd.read_csv("../Intrusion-Detection-on-NSL-KDD/a.csv")
# test_df = pd.read_csv("../Intrusion-Detection-on-NSL-KDD/b.csv")
train_df = pd.read_csv("train-train-norm.csv")
test_df = pd.read_csv("train-test-norm.csv")
# train_df = pd.read_csv("./nslkdd-train-pro.csv")
# test_df = pd.read_csv("./nslkdd-test-pro.csv")
# test_df.attack_map = test_df.attack_map.map(lambda _: np.random.randint(0, 5))

train_ds = data.NSLKDDDataset(train_df, False)
test_ds = data.NSLKDDDataset(test_df, False)

import json
# g_idxs_users = idx_split(len(train_df), USER_RATIOS)

##########################################################
# imbalance 
##########################################################
g_idxs_users = idx_split(len(train_df), USER_RATIOS)
for k in g_idxs_users.keys():
    g_idxs_users[k] = list(g_idxs_users[k])
json.dump(
    g_idxs_users,
    open("./res/FL/imbalance/g_idxs_users", "w")
)
json.dump(
    USER_RATIOS,
    open("./res/FL/imbalance/USER_RATIOS", "w")
)
# g_idxs_users = json.load(
#     open("./res/FL/imbalance/g_idxs_users", "r")
# )
# for i in range(NUM_USERS):
#     g_idxs_users[i] = list(g_idxs_users[str(i)])
# USER_RATIOS = json.load(
#     open("./res/FL/imbalance/USER_RATIOS", "r")
# )
##########################################################
# imbalance 
##########################################################



##########################################################
# non-IID
##########################################################
# g_idxs_users, USER_RATIOS = non_iid_ids_split(train_df)
# g_idxs_users = json.load(
#     open("./res/FL/non-IID/g_idxs_users", "r")
# )
# for i in range(4):
#     g_idxs_users[i] = list(g_idxs_users[str(i)])
# USER_RATIOS = json.load(
#     open("./res/FL/non-IID/USER_RATIOS", "r")
# )
##########################################################
# non-IID
##########################################################



print(f"[*] user_ratios: {USER_RATIOS}")
NUM_USERS = len(USER_RATIOS)

# train_ds, test_ds = data.get_datasets("./nslkdd-train-pro.csv", False, 0.1)
g_usr_train_dls = [
    DataLoader(
        DatasetSplit(train_ds, g_idxs_users[i]), 
        batch_size=LOCAL_BS, 
        shuffle=True
    )
    for i in range(NUM_USERS)
]
g_train_dl =  DataLoader(train_ds, TEST_BS, shuffle=True)
g_test_dl = DataLoader(test_ds, TEST_BS, shuffle=True)


g_net = NET

#初始化参数
def initial_g_net():
    for params in g_net.parameters():
        init.normal_(params, mean=0, std=0.01)

initial_g_net()
# ce_loss = torch.nn.CrossEntropyLoss()
#定义优化器
# optimizer = torch.optim.Adam(g_net.parameters(), lr=0.001,weight_decay=1e-4)
# opt_sgd = torch.optim.SGD(g_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
# for epoch in range(EPOCH):
#     print(f"[train] epoch {epoch}: ", end="")
#     train.train(g_net, ce_loss, opt_sgd, g_train_dl, DEVICE)
#     print(f"[test] epoch {epoch}: ", end="")
#     train.test(g_net, g_test_dl, DEVICE)

g_params = g_net.state_dict()
g_l_params = [g_params for _ in range(NUM_USERS)]
g_shape_info = em_client.extract_state_shape_info(g_params)
# for i in range(NUM_USERS):
#     g_l_nets[i].load_state_dict(g_params)

def local_train(l_net, l_ep, train_dl, device):
    t = time.time()
    l_net.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    opt_sgd = torch.optim.SGD(g_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)

    l_losses = []
    l_accs = []

    for le in range(l_ep):
        print(f"[train] local epoch {le}: ", end="")
        l_loss, l_acc = train.train(l_net, ce_loss, opt_sgd, train_dl, device )
        l_losses.append(l_loss)
        l_accs.append(l_acc)
    
    t = time.time() - t
    print(f"local train time: {t}")
        
    return sum(l_losses)/l_ep, sum(l_accs)/l_ep

def main_fed():
    global g_params
    global g_net
    train_losses = []
    train_metrics = []
    test_losses = []
    test_metrics = []
    l_usrs_losses = [[] for _ in range(NUM_USERS)]
    l_usrs_metrics = [[] for _ in range(NUM_USERS)]
    # import time
    for i in range(EPOCH):
        # t0 = time.time()
        l_losses = []
        l_accs = []
        m = max(int(FRAC * NUM_USERS), 1)
        choosed_usr_idxes = np.random.choice(range(NUM_USERS), m, replace=False)

        for usr_idx in choosed_usr_idxes:
            print(f"[usr {usr_idx}]:")
            # l_net = copy.deepcopy(g_net)
            # l_net = model.MLP(41, 48, 5) 
            # l_net = g_l_nets[usr_idx]
            l_net = g_net
            l_net.load_state_dict(g_params)
            
            l_loss, l_acc = local_train(l_net, LOCAL_EP, g_usr_train_dls[usr_idx], DEVICE)
            # l_loss, l_acc = local_train(l_net, LOCAL_EP, g_train_dl, DEVICE)
            l_params = l_net.state_dict()
            g_l_params[usr_idx] = copy.deepcopy(l_params)
            l_losses.append(copy.deepcopy(l_loss))
            l_accs.append(copy.deepcopy(l_acc))

        # g_params = FedAvg(g_l_params, USER_RATIOS)
        g_params = FedAvg(
            list(map(lambda i : g_l_params[i], choosed_usr_idxes)),
            list(map(lambda i : USER_RATIOS[i], choosed_usr_idxes))
        )
        # g_params = em_client.rFedAvg_enc(
        #     em_client.FLAS_ADDR,
        #     i, 
        #     list(map(lambda i : USER_RATIOS[i], choosed_usr_idxes)),
        #     list(map(lambda i : g_l_params[i], choosed_usr_idxes)),
        #     g_shape_info
        # )
        g_net.load_state_dict(g_params)

        g_avg_loss = sum(l_losses) / len(l_losses)
        print(f"Round {i:3d}, Average loss {g_avg_loss:.3f}")

        g_net.eval()
        train_loss, train_pred_stat = train.test(g_net, g_train_dl, DEVICE)
        test_loss, test_pred_stat = train.test(g_net, g_test_dl, DEVICE)
        train_metric = M.calc_metrics(train_pred_stat, 5)
        test_metric = M.calc_metrics(test_pred_stat, 5)
        print(f"[train dataset] Loss: {train_loss:.3f}, Acc: {100.0 * train_metric['acc']:.3f}%")
        print(f"[test  dataset] Loss: {test_loss:.3f}, Acc: {100.0 * test_metric['acc']:.3f}%")
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        test_losses.append(test_loss)
        test_metrics.append(test_metric)

        # calc g_net metrics on each usr's data
        for i in range(NUM_USERS):
            usr_dl = g_usr_train_dls[i]
            usr_loss, usr_pred_stat = train.test(g_net, usr_dl, DEVICE)
            usr_metric = M.calc_metrics(usr_pred_stat, 5)
            l_usrs_losses[i].append(usr_loss)
            l_usrs_metrics[i].append(usr_metric)

    return ((train_losses, train_metrics), (test_losses, test_metrics), (l_usrs_losses, l_usrs_metrics))


def client_train(train_dl):
    global g_net
    losses = []
    metrics = []
    ce_loss = torch.nn.CrossEntropyLoss()
    #定义优化器
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
    opt_sgd = torch.optim.SGD(g_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
    # for epoch in range(10):
    # for epoch in range(int(EPOCH * LOCAL_EP * FRAC)):
    for epoch in range(EPOCH):
        print(f"[train] epoch {epoch}: ", end="")
        train.train(g_net, ce_loss, opt_sgd, train_dl, DEVICE)
        print(f"[test] epoch {epoch}: ", end="")
        loss, pred_stat = train.test(g_net, g_test_dl, DEVICE)
        metric = M.calc_metrics(pred_stat, 5)
        print(f"loss: {loss:.3f}, acc: {metric['acc']:.3f}")
        losses.append(loss)
        metrics.append(metric)
    return losses, metrics

def main():
    client_train(g_train_dl)


CONF = {
    "num_users" : NUM_USERS,
    "user_ratios" : USER_RATIOS,
    "frac" : FRAC,
    "epoch" : EPOCH,
    "local_ep" : LOCAL_EP, # local epoch size
    "local_bs" : LOCAL_BS, # local batch size
    "test_bs" : TEST_BS,
    "model" : MODEL,
    "is_fl" : USE_FL
}
# # Compare between 
# if USE_FL:
#     _, (lsf, msf) = main_fed()
#     M.save_res(f"./res/FL/{MODEL}", CONF, lsf, msf)
# else:
#     ls, ms = main()
#     M.save_res(f"./res/ML/{MODEL}", CONF, ls, ms)

_, (lsf, msf), (luls, lums) = main_fed()

# xs = [i+ 1 for i in range(EPOCH)]
# luaccs = list(map(M.accs, lums))
import matplotlib.pyplot as plt

##########################################################
# non-IID
##########################################################

# M.save_res(f"./res/FL/non-IID", CONF, lsf, msf)

# # 不同客户端和 FL 之间的差别
# c_results = []
# for i in range(NUM_USERS):
#     cls, cms = client_train(g_usr_train_dls[i])
#     c_results.append((cls, cms))



# def plot_acc(xs, ys_fl, ys_c, title, save_path):
#     plt.figure()
#     plt.plot(xs, ys_fl, marker = "x", label="FL")
#     plt.plot(xs, ys_c, marker = "o", label="Local")
#     # plt.bar(xs, ys_c, width=0.8, bottom=None)
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.title(title)
#     plt.savefig(save_path)
#     # plt.show()

# xs = [i+ 1 for i in range(EPOCH)]

# for i in range(NUM_USERS):
#     plot_acc(xs, M.accs(msf), M.accs(c_results[i][1]), f"User{i}", f"./imgs/non-IID/user{i}_acc.png")

##########################################################
# non-IID
##########################################################


##########################################################
# imbalance 
##########################################################
# M.save_res(f"./res/FL/imbalance", CONF, lsf, msf)

# # 1. 不同客户端和 FL 之间的差别

# def plot_acc_imba(xs, ys_fl, ys_c, title, save_path):
#     plt.figure()
#     plt.plot(xs, ys_fl, marker = "x", label="FL")
#     plt.plot(xs, ys_c, marker = "o", label="Local")
#     # plt.bar(xs, ys_c, width=0.8, bottom=None)
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.title(title)
#     plt.savefig(save_path)
#     # plt.show()

# xs = [i+ 1 for i in range(EPOCH)]

# c_results = []
# for i in range(NUM_USERS):
#     cls, cms = client_train(g_usr_train_dls[i])
#     c_results.append((cls, cms))

# for i in range(NUM_USERS):
#     plot_acc_imba(xs, M.accs(msf), M.accs(c_results[i][1]), f"User{i}", f"./imgs/imbalance/user{i}_acc.png")

# # 2. 和集中式学习方法之间的比较
# M.plt_compare_figures("MLP", "imbalance")


##########################################################
# imbalance 
##########################################################


# plt.figure()
# plt.title("loss")
# plt.plot(xs, lsf, label="test")
# M.plt_usr_records(xs, luls)
# plt.savefig("usrs_losses.png")

# plt.figure()
# plt.title("loss")
# plt.plot(xs, lsf, label="test")
# M.plt_usr_records(xs, luls)
# plt.savefig("usrs_losses.png")

# plt.figure()
# plt.title("acc")
# plt.plot(xs, M.accs(msf), label="test")
# M.plt_usr_records(xs, luaccs)
# plt.savefig("usrs_accs.png")


# save_path = "./res/FL/non-IID/FL_acc.png"
# plt.figure()
# plt.title("FL non-IID acc")
# plt.xlabel("epoch")
# # plt.plot(xs, "", label = "ML")
# plt.plot(xs, M.accs(msf), label = "FL")
# plt.legend()
# plt.savefig(save_path)

# save_path = "./res/FL/non-IID/FL_loss.png"
# plt.figure()
# plt.title("FL non-IID loss")
# plt.xlabel("epoch")
# # plt.plot(xs, "", label = "ML")
# plt.plot(xs, lsf, label = "FL")
# plt.legend()
# plt.savefig(save_path)

# M.save_res(f"./res/FL_hyper/{MODEL}", CONF, lsf, msf)

# local_train(g_net, LOCAL_EP, g_train_dl, DEVICE)
# l_net = NET 
# l_net.load_state_dict(g_params)
# l_loss, l_acc = local_train(l_net, LOCAL_EP, g_train_dl, DEVICE)


# # num_epochs = 25
# list_acc = []

# ce_loss = torch.nn.CrossEntropyLoss()
# #定义优化器
# # optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
# opt_sgd = torch.optim.SGD(g_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
# for epoch in range(20):
#     print(f"[train] epoch {epoch}: ", end="")
#     train.train(g_net, ce_loss, opt_sgd, g_train_dl, DEVICE)
#     # print(f"[test] epoch {epoch}: ", end="")
#     # train.test(g_net, g_test_dl, DEVICE)