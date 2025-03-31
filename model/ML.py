
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
from torch import nn
class MLWrapper(nn.Module):
    def __init__(self, Obj = RandomForestClassifier):
        super().__init__()
        self.model = Obj()
    def fit(self, loader):
        x, y = loader.dataset.tensors
        x = np.array(x)
        y = np.array(y).squeeze(-1)
        self.model.fit(x, y)
    def forward(self, x):
        return torch.tensor(self.model.predict(np.array(x)))
    
