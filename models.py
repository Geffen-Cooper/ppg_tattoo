import torch
from torch import nn
import torch.quantization
import os
import torch.nn.functional as F

# ========================= Helper Functions =====================
def print_model_size(model):
    num_parameters = 0
    param_size = 0
    for param in model.parameters():
        num_parameters += param.nelement()
        param_size += param.nelement()*param.element_size()

    print("Number of parameters:", num_parameters)
    print("Model size (KB):",param_size/1e3)


class PPGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4,16,5)
        self.conv2 = nn.Conv1d(16,32,5)
        self.conv3 = nn.Conv1d(32,16,3,3)
        self.conv4 = nn.Conv1d(16,8,3)
        self.conv5 = nn.Conv1d(8,4,3,3)
        self.conv6 = nn.Conv1d(4,1,3)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x = F.relu(self.conv5(x))
        # print(x.shape)
        x = self.conv6(x)
        # print(x.shape)
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        return x
    

class PPGClassifierFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,2)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape)
        return x
    
if __name__ == '__main__':
    m = PPGClassifier()
    x = torch.rand(2,4,50)
    print(m(x))
    print_model_size(m)