# DATA620004
homework of DATA620004

## 1. 参数查找：学习率，隐藏层大小，正则化强度
三个参数分别有三种选择：学习率有[1, 0.5, 0.1]三种选择，隐藏层大小有[100, 200, 300]三种选择，正则化强度有[1e-5, 1e-4, 1e-3]三种选择，所以总共的组合有27种，选择在同等训练次数下最终达到的loss最小的组合作为最终的参数组合。（训练次数在当前文件中定义为200次，对应的文件为search_parm.py）

## 2. 训练

构建的类定义在search_parm.py中

### 2.1. 激活函数

在此次模型构建中,选择了三种不同的激活函数：sigmoid函数、ReLu函数、Leaky ReLu函数。具体表达式如下所示。
sigmoid函数：f(x) = 1 / (1 + exp(-x))
ReLu函数： $$f(x) = \max(0,x)$$
Leaky ReLu函数:  $$f(x) = \max(0.1x,x)$$
### 2.2. 反向传播，loss以及梯度的计算

loss函数采用交叉熵函数加上L2正则化的形式,定义在类中的loss_function函数。

关于梯度的计算，依次根据公式Downstream Gradient = Local Gradient*
Upstream Gradient反向求解，最终求得loss function关于两个权重矩阵w_1和w_2的偏导。在Classifier.py文件的myclassifier类中，定义了函数grad(self, n_sample, y_pred, h ,mu)用于计算。

### 2.3. 学习率下降策略

学习率初值设置如前文所示，在实验中固定迭代次数对学习率进行衰减，下降率为设为0.90，每隔100次迭代下降一次，具体实现在myclassifier类中的train函数。

### 2.4. L2正则化

正则化强度定义为\mu，代码中记为mu。

### 2.5.  优化器SGD

SGD是随机梯度下降，即随机选取初值进行梯度的计算和反向传播，这里定义选取和训练数据样本数一样大小的数据，具体实现在myclassifier类中的grad函数中。

### 2.6. 保存模型

训练后将得到的w_1和w_2保存在txt文件中，具体实现在myclassifier类中的save_data函数中。


#### 最终模型训练

利用参数查找得到的参数进行最后模型的训练，定义在plot.py文件中，最后绘制出模型训练和测试的loss曲线以及测试的accuracy曲线，并保存好训练的权重矩阵。



## 测试

运行model_display.py文件，导入模型输出分类的精度，并且绘制出权重矩阵w_1和w_2。





### 训练和测试步骤

首先运行Classifier.py文件，进行参数查找，最终得到的最优参数组合会存在子文件夹data，包括学习率、隐藏层大小、正则化强度以及此时选择的激活函数。再执行plot.py文件，读取前面选择好的参数，并进行模型训练，绘制出模型训练和测试的loss曲线以及测试的accuracy曲线，并保存好训练的权重矩阵在子文件夹data中。最后执行model_display.py文件，会利用保存的参数组合以及权重矩阵重新生成模型，对测试集进行测试，输出分类精度。最后该文件会可视化权重矩阵w_1和w_2。
