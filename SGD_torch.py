from model.model_torch import Model
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
def relative_error(y_true, y_pred):
    """计算相对误差"""
    return torch.norm(y_pred - y_true) / torch.norm(y_true)
#=========================Set Seed================================
torch.manual_seed(100)
#=========================Load Data================================
X_bias = torch.from_numpy(np.load('data/example1_D40_M10000/X_bias.npy')).float()
f_star = torch.from_numpy(np.load('data/example1_D40_M10000/f_star.npy')).float().reshape(-1, 1)
M = X_bias.shape[0]
split_index = int(M * 0.8)
X_train, X_test = X_bias[:split_index], X_bias[split_index:]
y_train, y_test = f_star[:split_index], f_star[split_index:]
#=========================Load Model===============================
D = X_bias.shape[1] - 1
m = 1000
model = Model(m, D)
#=======================Configure Training=========================
batch_size = 64
learning_rate = 0.2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss(reduction='mean')
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_losses = []
test_losses = []
relative_errors = []
#=======================Train Model=================================
for epoch in range(10000):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        # 打印参数L2范数
        # print(torch.norm(torch.cat([p.view(-1) for p in model.parameters()])))

    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_train = model(X_train)
        train_loss = loss_fn(y_pred_train, y_train)
        test_loss = loss_fn(y_pred_test, y_test)
        pred = model(X_bias)
        relative_error_ = relative_error(f_star, pred)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}, Relative Error: {relative_error_.item()}')
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        relative_errors.append(relative_error_.item())
#=======================Save Model==================================
is_save = True
if is_save:
    name = f'ModifiedNet_SGD_D{D}_{M}_lr{learning_rate}_{m}_torch'
    torch.save(model.state_dict(), f"save/{name}.pth")
    # 保存为 JSON 文件
    history_dict = {
        'train_loss': train_losses,
        'test_loss': test_losses, 
        'relative_error': relative_errors
    }
    with open(f'save/{name}_history.json', 'w') as f:
        json.dump(history_dict, f)