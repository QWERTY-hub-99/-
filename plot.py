# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 09:41:38 2023

@author: cym
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from Classifier import myclassifier

PATH = 'C:/Users/cym/Desktop/network_hw1/'

# 加载模型


D_in = 784
D_out = 10


f = open(PATH + 'data/activation.txt' , 'rb')
activation = pickle.load(f)
f.close()
f = open(PATH + 'data/' + activation + '_hidden_nodes.txt', 'rb')
hidden_nodes = pickle.load(f)
f.close()
f = open(PATH + 'data/' + activation + '_mu.txt', 'rb')
mu = pickle.load(f)
f.close()
f = open(PATH + 'data/' + activation + '_learning_rate.txt', 'rb')
lr = pickle.load(f)
f.close()


model = myclassifier(D_in, hidden_nodes, D_out, activation)


test_flag = 1  # 等于1时表示每次迭代训练都进行测试集的测试并保存结果
number_of_train = 400  # 训练次数
print_flag = 1  # 等于1时打印训练的loss

# 训练模型
model.train(number_of_train, mu, lr, test_flag, print_flag)
# 可视化训练和测试的loss曲线
x = np.arange(len(model.train_loss))
plt.plot(x, model.train_loss, label='train loss')
plt.plot(x, model.test_loss, label='test loss', linestyle='--')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(activation + '_loss_image.jpg', dpi = 300)
plt.show()

# 可视化测试的accuracy曲线
plt.plot(x, model.test_accuracy, label='test accuracy')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig(activation + '_accuracy_image.jpg', dpi = 300)
plt.show()
print("测试集的分类精度：", model.test(0, mu))  # 输出测试集上的结果

# 保存模型的权重矩阵
model.save_data()


