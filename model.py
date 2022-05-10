import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(dim_in, dim_hidden)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(dim_hidden, 2 * dim_hidden)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.dense3 = nn.Linear(2 * dim_hidden, dim_out)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        return x

class GRU_MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(GRU_MLP, self).__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        # self.hidden_dim = 128 
        # self.n_layer = 1
        self.gru = nn.GRU(self.in_dim, 128, 1)
        self.dense1 = nn.Linear(128, 48)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(48, 48)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(48, n_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x = x.permute(1, 0, 2)
        out, _ = self.gru(x)
        x = out[-1, :, :]
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.classifier(x)
        return x


class GRU_MLP2(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(GRU_MLP2, self).__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.gru = nn.GRU(self.in_dim, 48, 1)
        self.dense1 = nn.Linear(48, 48)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(48, 48)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(48, n_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x = x.permute(1, 0, 2)
        out, _ = self.gru(x)
        x = out[-1, :, :]
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.classifier(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_dim, n_classes) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        conv1_out_channel = 16
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, conv1_out_channel, 3, padding_mode="replicate"),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.dense1 = nn.Linear(conv1_out_channel * 19, 128)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        return x
