Note: 请使用pytorch框架，尽量不要使用torch中的自带优化器

### 模型结构

令$\theta = (\mathbf{W}, \vec{a})$现在有神经网络模型， X已经用1增广，没有偏置:

$$
f(\vec{x}, \theta)=\frac{1}{m}\sum_{i=1}^m a_i \cdot relu(W_i^T \vec{x})
$$

其中，m为隐藏层维度，输入维度X为(D + 1)，所以参数W的形状为 $(D+1)*m$ ,a 为 $m*1$ 的向量，输出为scalar。

定义损失函数I为：

$$
I = \int_{R^D} (f(\theta) - f^*)^2 d\vec{x} := \frac{1}{N}\sum_{i=1}^N {(f - f^*)^2}
$$

### 数据集

定义目标函数：

$$
f(x_1, \cdots, x_D) = \sin(\sum_{i=1}^D p_ix_i) + \cos(\sum_{i=1}^D q_ix_i)
$$

${𝑝𝑖,𝑞𝑖}_{1≤𝑖≤𝐷}$ 请你生成数据集 ${𝐱_𝑗,𝑓∗_𝑗 }_{1≤𝑗≤𝑀}$, 其中$𝐱_𝑗∈(0, 1)_𝐷$

请用Z-score normalization预处理数据,数据集要求D=20,生成M条，M=10000。

在训练时，请使用mini-batch技术，定义batch size为256；训练集和测试集为8:2

### 算法

将机器学习优化问题看作如下框架：

$$
\frac{d\theta}{dt} = -\frac{\delta I(\theta)}{\delta \theta} := -N(\theta)
$$

在每一个epoch中，进行如下更新：

输入:

初始参数: $\theta_0$ 

时间步长: $\Delta t$ 

标量辅助变量初值: $r_0 = \sqrt{I(\theta_0) + C}$

具体步骤：

1.	初始化: 设定初始参数$\theta_0$和初始标量变量$r_0$ 。

2.	循环迭代 (从  $n = 0$  到  $N - 1$ )：
	
    步骤 1: 计算中间参数$\theta_{n+1,1} = \theta_n$ 。
	步骤 2: 更新第二个中间参数$\theta_{n+1,2}$ ，计算公式为：
    $$
    \theta_{n+1,2} = -\Delta t \sqrt{\mathcal{I}(\theta_n) + C} \cdot (I + \Delta t \mathcal{L})^{-1} \mathcal{N}(\theta_n)
    $$
    步骤 3: 更新标量辅助变量 $r_{n+1}$ 
    $$
    r_{n+1} = \frac{r_n}{1 + \Delta t \cdot \frac{(\mathcal{N}(\theta_n), (I + \Delta t \mathcal{L})^{-1} \mathcal{N}(\theta_n))}{2(\mathcal{I}(\theta_n) + C)}}
    $$
    步骤 4: 更新参数：
    $$
    \theta_{n+1} = \theta_{n+1,1} + r_{n+1} \cdot \theta_{n+1,2}
    $$

3.	返回: 最终的参数  $\theta_N$ 。