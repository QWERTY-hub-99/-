# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:45:07 2023

@author: cym
"""

import matplotlib.pyplot as plt
from Classifier import myclassifier
import pickle

PATH = 'C:/Users/cym/Desktop/network_hw1/'

# 加载模型所需参数

f = open(PATH + 'data/activation.txt' , 'rb')
activation = pickle.load(f)
f.close()
f = open(PATH + 'data/' + activation + '_hidden_nodes.txt', 'rb')
hidden_nodes = pickle.load(f)
f.close()
f = open(PATH + 'data/' + activation + '_mu.txt', 'rb')
mu = pickle.load(f)
f.close()

# 构建模型
D_in = 784
D_out = 10
final_model = myclassifier(D_in, hidden_nodes, D_out, activation)

# 进行测试，输出分类精度
print("测试集的分类精度：", final_model.test(0, mu))

# 可视化每层的网络参数
figure = plt.figure()
axes = figure.add_subplot(221)
caxes = axes.matshow(final_model.w1[0:200, :].T, interpolation='nearest')
figure.colorbar(caxes)
axes = figure.add_subplot(222)
caxes = axes.matshow(final_model.w1[200:400, :].T, interpolation='nearest')
figure.colorbar(caxes)
axes = figure.add_subplot(223)
caxes = axes.matshow(final_model.w1[400:600, :].T, interpolation='nearest')
figure.colorbar(caxes)
axes = figure.add_subplot(224)
caxes = axes.matshow(final_model.w1[600:784, :].T, interpolation='nearest')
figure.colorbar(caxes)
plt.savefig(activation + '_w1_image_break.jpg', dpi = 300)
plt.show()

plt.matshow(final_model.w1.T)
plt.colorbar()
plt.savefig(activation + '_w1.jpg', dpi = 300)
plt.show()

plt.matshow(final_model.w2.T)
plt.colorbar()
plt.savefig(activation + '_w2.jpg', dpi = 300)
plt.show()


