"""
System Environment
OS: Windows 10 家庭中文版
CPU: intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz 2.80GHz
GPU: GeForce GTX 1050
CUDA 9.1.84
Memory: 16 GiB

Python Environment
python 3.7.4 (anaconda)
numpy 1.16.5 
pandas 0.25.1
pytorch 1.1.0
scikit-learn 0.21.3
scipy 1.3.1
matplotlib 3.1.1
"""
"""
created: 2021/1/3
training & predicting part copied from TA's code.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from collections import OrderedDict
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import Normalizer

from sklearn.feature_selection import SelectKBest

from utils import numout2boolout

# 降维处理，以训练集为标准
def dimension_reduction(X, x, save_path_train, save_path_test):
    """
    dimension reduction with PCA.
    input:
    X: array(n_arg_train, n_features), train set data to be reduced
    x: array(n_arg_test, n_features), test set data to be reduced
    save_path_train/test: str, path of the reduced data(array(n_arg, n_feature_reduced)) to be saved
    output: none
    """
    reductor = PCA(n_components=10)

    X_trans = reductor.fit_transform(X)
    x_trans = reductor.transform(x)

    np.save(save_path_train, X_trans)
    np.save(save_path_test, x_trans)


# 一个简单的神经网络(2个隐藏层)
class simple_NN(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2):
        super(simple_NN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, 2)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.tanh(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x


def main():
    learning_rate = 1e-3
    weight_decay = 1e-3
    epoches = 300
    log_interval = 10

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    Y = numout2boolout(np.load("./processedData/Y/Y_train.npy"))
    y = numout2boolout(np.load("./processedData/Y/y_test.npy"))

    weights_of_lable = np.zeros(2)
    print("# of labels in Y:")
    for i in range(2):
        num_label = len(Y[Y==i])
        print(i, num_label)
        weights_of_lable[i] = 1 / num_label

    weights = [weights_of_lable[int(i)] for i in Y]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    X = np.load("./processedData/X_train_masked.npy")
    x = np.load("./processedData/x_test_masked.npy")
    #X = np.load("./processedData/X/X_train_reduced.npy")
    #x = np.load("./processedData/X/x_test_reduced.npy")

    # 对数据进行特征选择处理
    skb = SelectKBest(k=10)
    X_trans = skb.fit_transform(X, Y)
    x_trans = skb.transform(x)

    normalizer = Normalizer()
    X_trans = normalizer.fit_transform(X_trans)
    x_trans = normalizer.transform(x_trans)

    train_data = TensorDataset(torch.tensor(X_trans, dtype=torch.float), torch.tensor(Y, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(x_trans, dtype=torch.float), torch.tensor(y, dtype=torch.long))
    train_data_loader = DataLoader(train_data, batch_size=16, shuffle=True)#sampler=sampler)
    test_data_loader = DataLoader(test_data, batch_size=8, shuffle=True)

    # 构造模型
    model = simple_NN(np.shape(X)[1], 8, 8)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)
    #criterion = nn.NLLLoss(reduction='sum')
    criterion = nn.CrossEntropyLoss()

    # 训练
    model.train()
    print("Training...")
    for epoch in range(epoches):
        loss_sum = 0.
        for data, label in train_data_loader:
            #data, label = data.to(DEVICE), label.to(DEVICE)
            # Forward
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss_sum += loss
            # Backward
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % log_interval == 0:
            print("Epoch = {0}, loss = {1:.5f}".format(
                epoch + 1, loss.data.float()))
    
    # 测试
    # 这一段是为了观察是否有收敛
    model.eval()
    print("Predicting...")
    with torch.no_grad():
        labels, pred = np.array([]), np.array([])
        loss_sum = 0.
        print('result in train')
        for data, label in train_data_loader:
            #data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data)
            category = np.argmax(out, axis=1)#.cpu(), axis=1)
            loss = criterion(out, label)
            loss_sum += loss#.cpu()
            labels = np.append(labels, label)#.cpu())
            pred = np.append(pred, category)

        labels_trans = labels
        pred_trans = pred

        acc = accuracy_score(labels_trans, pred_trans)

        cm = confusion_matrix(labels_trans, pred_trans)
        precision = precision_score(labels_trans, pred_trans, average='micro')
        recall = recall_score(labels_trans, pred_trans, average='micro')

        print("Test loss = {0:.5f}".format(loss_sum))
        print("Test accuracy = {0:.5f}".format(acc))
        np.set_printoptions(precision=5)
        print('Test cm = ')
        print(cm)
        print('Test precision = {0:.5f}'.format(precision))
        print('Test recall = {0:.5f}'.format(recall))

    # 测试集上的结果
    with torch.no_grad():
        labels, pred = np.array([]), np.array([])
        loss_sum = 0.
        print('result in test')
        for data, label in test_data_loader:
            #data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data)
            category = np.argmax(out, axis=1)#.cpu(), axis=1)
            loss = criterion(out, label)
            loss_sum += loss#.cpu()
            labels = np.append(labels, label)#.cpu())
            pred = np.append(pred, category)

        labels_trans = labels
        pred_trans = pred

        acc = accuracy_score(labels_trans, pred_trans)

        cm = confusion_matrix(labels_trans, pred_trans)
        precision = precision_score(labels_trans, pred_trans, average='micro')
        recall = recall_score(labels_trans, pred_trans, average='micro')

        print("Test loss = {0:.5f}".format(loss_sum))
        print("Test accuracy = {0:.5f}".format(acc))
        np.set_printoptions(precision=5)
        print('Test cm = ')
        print(cm)
        print('Test precision = {0:.5f}'.format(precision))
        print('Test recall = {0:.5f}'.format(recall))
    
    torch.save(model.state_dict(), "./NN_model/model1.pt")

if __name__ == "__main__":
    main()
