# -*- coding: utf-8 -*-
'''
Created on Wed Apr  5 17:30:20 2023

@author: cym
'''

import numpy as np
import os
import struct
import pickle

PATH = 'C:/Users/cym/Desktop/network_hw1/'

class myclassifier:
    
    def __init__(self , D_in , H , D_out, Activation = 'leaky_relu'):
        # D_in size of x input
        # H  size of hide
        # D_out size of output
        # Activation choice of Activation function
        
        self.inputsize = D_in
        self.hidesize = H
        self.outsize = D_out
        self.Activation = Activation
        self.train_loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_data()
        
        #weight matrix
        self.w1 = np.random.randn(self.inputsize , self.hidesize)
        self.w2 = np.random.randn(self.hidesize , self.outsize)
        
        if os.path.exists(PATH + 'data/' + self.Activation + '_w1.txt'):
            self.w1 = np.loadtxt(PATH + 'data/' + self.Activation + '_w1.txt')
        else:
            pass

        if os.path.exists(PATH + 'data/' + self.Activation + '_w2.txt'):
            self.w2 = np.loadtxt(PATH + 'data/' + self.Activation + '_w2.txt')
        else:
            pass
    
    def load_data(self):
        X_test, Y_test = self.load_mnist('mnist', 't10k')
        X_test = X_test / 255 * 0.99 + 0.01
        Y_test = self.convert_to_one_hot(Y_test, 10)
        X_train, Y_train = self.load_mnist('mnist', 'train')
        Y_train = self.convert_to_one_hot(Y_train, 10)
        X_train = X_train / 255 * 0.99 + 0.01
        
        return X_train, Y_train, X_test, Y_test
        
    @staticmethod
    def load_mnist(path, kind='train'):
        '''Load MNIST data from `path`'''
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II' , lbpath.read(8))
            labels = np.fromfile(lbpath , dtype=np.uint8)
    
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII' , imgpath.read(16))
            images = np.fromfile(imgpath , dtype=np.uint8).reshape(len(labels), 784)
            
        return images, labels

    @staticmethod
    def convert_to_one_hot(y,c):
        #convert labels to one-hot array
        return np.eye(c)[y.reshape(-1)]
    
    def forward_pass(self, x):
        temp = x @ self.w1
        
        if self.Activation == 'leaky_relu':
            h = np.where(temp > 0, temp, temp * 0.01) # leaky relu
            
        elif self.Activation == 'relu':
            h = (temp + abs(temp)) / 2 # ReLu
            
        elif self.Activation == 'sigmoid':
            h = 1 / (1 + np.exp(-temp)) #sigmoid
        
        y_pred = h @ self.w2
        y_pred = self.softmax(y_pred)
        
        return y_pred, h
    
    @staticmethod
    def softmax(s):
        s_max = s - s.max(axis=1).reshape((-1, 1))
        temp_s = np.exp(s_max)
        temp = temp_s / temp_s.sum(axis = 1).reshape((-1,1))        
        
        return temp
        
    def loss_function(self, mu, y_pred, y):
        loss = self.cross_entropy_error(y_pred, y) + 0.5 * mu * np.sum(
            self.w1 * self.w1) + 0.5 * mu * np.sum(self.w2 * self.w2)
        
        return loss
    
    @staticmethod
    def cross_entropy_error(y_pred, y):
        error = - np.sum(y * np.log(y_pred + 1e-8))
        
        return error
    
    @staticmethod
    def get_label(y):
        label = np.argmax(y, axis = -1)
        
        return label
    
    def grad(self, n_sample, y_pred, h ,mu):
        sample = np.random.randint(0, self.X_train.shape[0], n_sample)
        y_pred_temp = y_pred[sample, :]
        y_temp = self.Y_train[sample, :]
        h_temp = h[sample, :]
        x_temp = self.X_train[sample, :]
        dy_pred_pre = y_pred_temp - y_temp
        
        dw2 = h_temp.T @ dy_pred_pre + mu * np.ones(self.w2.shape)
        #dw2 = h_temp.T @ dy_pred_pre + mu * self.w2
        
        dh = dy_pred_pre @ self.w2.T
        
        if self.Activation == 'leaky_relu':
            dw1 = x_temp.T @ (dh * (np.where(h_temp > 0, 1, 0.01))) # leaky relu
            
        elif self.Activation == 'relu':
            dw1 = x_temp.T @ (dh * (np.int64(h_temp > 0))) # relu
            
        elif self.Activation == 'sigmoid':
            dw1 = x_temp.T @ (dh * (h_temp * (1 - h_temp))) # sigmoid
        
        dw1 += mu * np.ones(self.w1.shape)
        #dw1 += mu * self.w1
        
        return dw1, dw2
    
    def train(self, number_of_train, mu, lr, test_flag, print_flag):
        for i in range(number_of_train):
            #forward pass & calculate loss and record
            y_pred, h = self.forward_pass(self.X_train)
            loss = self.loss_function(mu, y_pred, self.Y_train)
            self.train_loss.append(loss)
            
            if print_flag == 1: # print loss
                print('The loss of the %d train is %10.8f' %(i+1, loss))
            else:
                pass
            
            #update w1 & w2
            n_sample = 128
            dw1, dw2 = self.grad(n_sample, y_pred, h, mu)
            
            self.w1 -= lr * (dw1 / n_sample)
            self.w2 -= lr * (dw2 / n_sample)
            
            #lr decrease per 100 iterations; 0.9 is the decrease factor
            if i % 100 == 0:
                lr = lr * 0.9
            
            if test_flag == 1: # the test set is tested once per iteration
                self.test(1, mu)
            else:
                pass
            
    def test(self, test_flag, mu):
        temp = self.X_test @ self.w1
        
        if self.Activation == 'leaky_relu':
            h = np.where(temp > 0, temp, temp * 0.01) # leaky relu
            
        elif self.Activation == 'relu':
            h = (temp + abs(temp)) / 2 # ReLu
            
        elif self.Activation == 'sigmoid':
            h = 1 / (1 + np.exp(-temp)) #sigmoid
        
        y_pred = h @ self.w2
        y_pred = self.softmax(y_pred)
        
        accuracy = np.sum(self.get_label(y_pred) == self.get_label(self.Y_test)) / self.Y_test.shape[0]
        
        if test_flag == 1:
            self.test_accuracy.append(accuracy)
            loss = self.loss_function(mu, y_pred, self.Y_test)
            self.test_loss.append(loss)
        else:
            pass
        
        return accuracy
    
    def save_data(self): #save the data of w1 and w2 in txt file
        name_w1 = PATH + 'data/' + self.Activation + '_w1.txt'
        name_w2 = PATH + 'data/' + self.Activation + '_w2.txt'
        with open(name_w1, 'w') as ww1:
            np.savetxt(ww1, self.w1)
        with open(name_w2, 'w') as ww2:
            np.savetxt(ww2, self.w2)

if __name__ == '__main__':
    
    D_in = 784
    D_out = 10
    H = [100, 200, 300]
    mu = [1e-5, 1e-4, 1e-3]
    lr = [1, 0.5, 0.1]
    test_flag = 0
    number_of_train = 200
    print_flag = 1
    loss_min = np.inf
    
    H1 = H[0]
    mu1 = mu[0]
    lr1 = lr[0]
    
    activation = 'sigmoid'#'leaky_relu'#'relu'
    
    for H_i in H:
        for mu_i in mu:
            for lr_i in lr:
                model = myclassifier(D_in, H_i, D_out, activation)
                model.train(number_of_train, mu_i, lr_i, test_flag, print_flag)
                if model.train_loss[-1] < loss_min:
                    loss_min = model.train_loss[-1]
                    print('Now minimum of train loss is' + str(loss_min))
                    H1 = H_i
                    mu1 = mu_i
                    lr1 = lr_i
                else:
                    pass
                
    print('The final lr is ', lr1, '; The final hidden size is ', H1, '; The final normalization parameter is ', mu1)
    
    # 保存各项参数
    f = open(PATH + 'data/' + activation + '_hidden_nodes.txt', 'wb')
    pickle.dump(H1, f)
    f.close()
    f = open(PATH + 'data/' + activation + '_mu.txt', 'wb')
    pickle.dump(mu1, f)
    f.close()
    f = open(PATH + 'data/' + activation + '_learning_rate.txt', 'wb')
    pickle.dump(lr1, f)
    f.close()
    f = open(PATH + 'data/activation.txt', 'wb')
    pickle.dump(activation, f)
    f.close()
    