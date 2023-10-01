# import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset
class NSLKDDDataset(Dataset):
    def __init__(self, df, is_binary):
        # df = pd.read_csv(file_name)
        
        x = df.iloc[:, :41].values
        # scaler = Normalizer().fit(x)
        # x = scaler.transform(x)

        if is_binary:
            y = df.is_attack.values # 二分类
            self.classes = 2
        else:
            y = df.attack_map.values # 五分类
            self.classes = 5
        
        # y = np.array([4 for i in range(len(y))])

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]