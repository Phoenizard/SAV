import torch
import torch.nn as nn

class DivideByConstantLayer(nn.Module):
    def __init__(self, constant):
        super(DivideByConstantLayer, self).__init__()
        self.constant = constant

    def forward(self, x):
        return x / self.constant
    
class Model(nn.Module):
    def __init__(self, m, D):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(D + 1, m)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(m, 1)
        self.divide = DivideByConstantLayer(m)

        # HE initialization
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        
        
    def forward(self, x):
        fc1 = self.linear1(x)
        fc2 = self.relu(fc1)
        fc3 = self.linear2(fc2)
        fc4 = self.divide(fc3)
        return fc4
