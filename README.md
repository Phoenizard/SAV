## SAV Project

这个代码库包含了SGD和SAV两种算法，主要程序就是SGD_torch.py和SAV.py，在运行时，请配置：

torch, numpy, json

```
python SGD_torch.py
python SAV.py
```

### 模型定义

模型结构在 model/model_torch.py 的文件中，其中Model是主要模型结构:

- forward 函数包含了向前传播的过程，模仿毛老师代码的话需要尝试不同tanh+hidden的组合结构。

- self.div(x) = DivideByConstantLayer(m) 的含义是，对x张量每个元素都除以m

在修改好模型代码后运行SGD/SAV即可自动调整

### 参数设置

数据集：我预设了四组data,在data文件夹下可以看到四个example的文件夹，其中D后面代表了样本的维度，M代表有多少样本数，都已经做过标准化了。

在使用时请关注SGD和SAV文件中的Load Data部分：
X_bias 和 f_star 都需要输入一个路径，请选择合适的数据路径填写就好。

#### SGD

- Line 20: m为隐藏层大小
- Line 23: batch_size 批量大小
- optim.SGD中lr对应学习率

#### SAV

基本和SGD一致，C，_lambda = SAV中的C和线性算子L

### 结果

结果会展现在终端中，如果觉得结果还不错，可以把代码中的is_save改为True，并且修改name为你想要的名字。结果将自动保存在 save 文件夹下，一个json文件记录了损失函数的变化，还有一个.pth记录了最后的模型参数。

如果需要绘制损失函数，请打开load_loss并把第5行的路径修改为新保存的json文件即可。

如果需要查看模型结果（绘图仅支持1维数据），请打开load_model_torch并且修改18行 pth的路径 20，21数据集的路径；就可以查看原函数和拟合的函数效果图