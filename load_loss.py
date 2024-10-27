import matplotlib.pyplot as plt
import json

# 加载训练历史
with open('save/SAV_D40_10000_lr02_1000_torch_PaperData_history.json', 'r') as f:
    history_dict = json.load(f)

# 绘制训练和验证损失曲线
plt.plot(history_dict['train_loss'], label='Training loss')
plt.plot(history_dict['test_loss'], label='Validation loss')
plt.plot(history_dict['relative_error'], label='Relative error')
plt.grid(True)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()