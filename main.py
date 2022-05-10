import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import init

import data
import model
import train

BATCH_SIZE = 32 
EPOCH = 20
DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# train_df = pd.read_csv("../Intrusion-Detection-on-NSL-KDD/a.csv")
# test_df = pd.read_csv("../Intrusion-Detection-on-NSL-KDD/b.csv")
train_df = pd.read_csv("train-train.csv")
test_df = pd.read_csv("train-test.csv")
# train_df = pd.read_csv("./nslkdd-train-pro.csv")
# test_df = pd.read_csv("./nslkdd-test-pro.csv")
# test_df.attack_map = test_df.attack_map.map(lambda _: np.random.randint(0, 5))

train_ds = data.NSLKDDDataset(train_df, False)
test_ds = data.NSLKDDDataset(test_df, False)

# train_ds, test_ds = data.get_datasets("./nslkdd-train-pro.csv", False, 0.1)
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=True)

# net = model.MLP(41, 48, 5)
# GRU
# net = model.Rnn(28, 10, 2, 10)
net = model.GRU_MLP(41, 5)
net = net.to(DEVICE)

#初始化参数
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# num_epochs = 25
list_acc = []

ce_loss = torch.nn.CrossEntropyLoss()
#定义优化器
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
opt_sgd = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
for epoch in range(EPOCH):
    print(f"[train] epoch {epoch}: ", end="")
    train.train(net, ce_loss, opt_sgd, train_dl, DEVICE)
    print(f"[test] epoch {epoch}: ", end="")
    train.test(net, test_dl, DEVICE)