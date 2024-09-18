from model.model_torch import Model
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
#=========================Set Seed================================
torch.manual_seed(100)
#=========================Load Data================================
X_bias = torch.from_numpy(np.load('data/example1_D1_M10000/X_bias.npy')).float()
f_star = torch.from_numpy(np.load('data/example1_D1_M10000/f_star.npy')).float().reshape(-1, 1)
M = X_bias.shape[0]
split_index = int(M * 0.8)
X_train, X_test = X_bias[:split_index], X_bias[split_index:]
y_train, y_test = f_star[:split_index], f_star[split_index:]
#=========================Load Model===============================
D = X_bias.shape[1] - 1
m = 50
model = Model(m, D)
#=======================Configure Training=========================
batch_size = 64
learning_rate = 1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction='mean')
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#=======================Train Model=================================
for epoch in range(10000):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        # print(y_pred.shape, y_batch.shape)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
#=======================Save Model==================================
is_save = True
if is_save:
    name = f'SGD_D{D}_{M}_1_{m}_torch'
    torch.save(model.state_dict(), f"save/{name}.pth")
    # 保存为 JSON 文件
    history_dict = {'loss': loss.item()}
    with open(f'save/{name}_history.json', 'w') as f:
        json.dump(history_dict, f)