import matplotlib.pyplot as plt
import json

# 加载训练历史
with open('save/SGD_D1_64_2e1_history.json', 'r') as f:
    history_dict = json.load(f)

# 绘制训练和验证损失曲线
plt.plot(history_dict['loss'], label='Training loss')
plt.plot(history_dict['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()