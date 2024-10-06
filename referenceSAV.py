from model.model_torch import Model
import torch
from torch.utils.data import DataLoader, TensorDataset
from sav_opt import SAV
import numpy as np
import json
def relative_error(y_true, y_pred):
    """计算相对误差"""
    return torch.norm(y_pred - y_true) / torch.norm(y_true)
#=========================Set Seed================================
torch.manual_seed(100)
#=========================Load Data================================
X_bias = torch.from_numpy(np.load('data/example1_D1_M200/X_bias.npy')).float()
f_star = torch.from_numpy(np.load('data/example1_D1_M200/f_star.npy')).float().reshape(-1, 1)
M = X_bias.shape[0]
split_index = int(M * 0.8)
X_train, X_test = X_bias[:split_index], X_bias[split_index:]
y_train, y_test = f_star[:split_index], f_star[split_index:]
#=========================Load Model===============================
D = X_bias.shape[1] - 1
m = 50
model = Model(m, D)
#=======================Configure Training=========================
C = 1
_lambda = 0
batch_size = 32
learning_rate = 10
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = SAV(model.parameters(), lr=learning_rate, const=C, weight_decay=_lambda)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#=======================Train Model=================================
for epoch in range(100000):
    for X_batch, y_batch in train_loader:
        def closure():
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            return loss #, outputs
        loss = optimizer.step(closure)
    with torch.no_grad():
        # 计算训练集误差
        y_pred_train = model(X_train)
        train_loss = loss_fn(y_pred_train, y_train)
        # 计算测试集误差
        y_pred_test = model(X_test)
        test_loss = loss_fn(y_pred_test, y_test)
        # 计算相对误差
        pred = model(X_bias)
        relative_error_ = relative_error(f_star, pred)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}, Relative Error: {relative_error_.item()}')
        if relative_error_ < 0.0001:
            break
#=======================Save Model==================================
is_save = True
if is_save:
    name = f'Ref_SAV_D{D}_{M}_lr10_{m}_torch'
    torch.save(model.state_dict(), f"save/{name}.pth")
    # 保存为 JSON 文件
    history_dict = {'loss': loss.item()}
    with open(f'save/{name}_history.json', 'w') as f:
        json.dump(history_dict, f)