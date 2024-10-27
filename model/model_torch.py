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
        self.linear1 = torch.nn.Linear(D + 1, m, bias=False)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(m, 1, bias=False)
        self.div = DivideByConstantLayer(m)
        self.hidden = nn.Linear(m, m)
        # 初始化
        self.init_weights()

    def init_weights(self):
        # 使用He初始化对线性层进行权重初始化
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='tanh')
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='tanh')
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='tanh')

    def forward(self, x):
        fc1 = self.linear1(x)
        fc2 = self.tanh(fc1)
        fc3 = self.hidden(fc2)
        fc4 = self.tanh(fc3)
        fc5 = self.hidden(fc4)
        fc6 = self.tanh(fc5)
        fc7 = self.hidden(fc6)
        fc8 = self.tanh(fc7)
        fc9 = self.hidden(fc8)
        fc10 = self.tanh(fc9)
        fc11 = self.hidden(fc10)
        fc12 = self.linear2(fc11)
        fc13 = self.div(fc12)
        return fc13
    
        