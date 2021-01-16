"""
created: 2021/1/3
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

from header import numout2boolout

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
        self.layer1 = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(n_in, n_hidden1)),
                    ("activation1", nn.Sigmoid())
                ]
            )
        )
        self.layer2 = nn.Sequential(
            OrderedDict(
                [
                    ("fc2", nn.Linear(n_hidden1, n_hidden2)),
                    ("activation2", nn.Sigmoid())
                ]
            )
        )
        self.out_layer = nn.Sequential(
            OrderedDict(
                [
                    ("fc2", nn.Linear(n_hidden2, 4)),
                    ("output", nn.LogSoftmax(dim=1))
                ]
            )
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out_layer(x)
        return x


def main():
    learning_rate = 1e-3
    weight_decay = 1e-5
    epoches = 100
    log_interval = 10

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    Y = np.load("./processedData/Y_train.npy") - 1
    y = np.load("./processedData/y_test.npy") - 1
    Xx = np.load("./processedData/Xx.npy")
    X = Xx[:len(Y), :]
    x = Xx[len(Y):, :]

    normalizer = Normalizer()
    X_trans = normalizer.fit_transform(X)
    x_trans = normalizer.transform(x)

    train_data = TensorDataset(torch.tensor(X_trans, dtype=torch.float), torch.tensor(Y, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(x_trans, dtype=torch.float), torch.tensor(y, dtype=torch.long))
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # 构造模型
    model = simple_NN(np.shape(X)[1], 20, 20).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)
    criterion = nn.NLLLoss(reduction='sum')
    scheduler = StepLR(optimizer, 10, 0.9)

    # 训练
    model.train()
    print("Training...")
    for epoch in range(epoches):
        loss_sum = 0.
        for data, label in train_data_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            # Forward
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss_sum += loss
            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        if (epoch + 1) % log_interval == 0:
            print("Epoch = {0}, loss = {1:.5f}".format(
                epoch + 1, loss.data.float()))
    
    # 测试
    model.eval()
    print("Predicting...")
    with torch.no_grad():
        labels, pred = np.array([]), np.array([])
        loss_sum = 0.
        for data, label in test_data_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data)
            category = np.argmax(out.cpu(), axis=1)
            loss = criterion(out, label)
            loss_sum += loss.cpu()
            labels = np.append(labels, label.cpu())
            pred = np.append(pred, category)

        labels_trans = numout2boolout(labels+1)
        pred_trans = numout2boolout(pred+1)

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
