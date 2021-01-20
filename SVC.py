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
'''
使用交叉验证方法测试不同核函数以及不同正则化强度的效果
'''
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from utils import PUK_kernel,Y,y,numout2boolout,reactantCombination

# 交叉验证方法
def CV(X, Y, Model, params, n_splits=3, numlabel=True, shuffle=True):
    X_std = StandardScaler().fit_transform(X)
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    rec = 0
    acc = 0
    prec = 0
    for train_index_rc, test_index_rc in kf.split(reactantCombination):
        train_index = [
            i for rc in train_index_rc for i in reactantCombination[rc]]
        test_index = [
            i for rc in test_index_rc for i in reactantCombination[rc]]
        X_train, X_test = X_std[train_index], X_std[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model = Model(**params)
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        if numlabel:
            Y_test = numout2boolout(Y_test)
            pred = numout2boolout(pred)
        rec += recall_score(Y_test, pred, average='weighted')
        prec += precision_score(Y_test, pred, average="weighted")
        acc += accuracy_score(Y_test, pred)
    return [rec/n_splits, prec/n_splits, acc/n_splits]

X_masked = np.load("./processedData/X/X_masked.npy")
kernelName=["PUK","rbf","sigmoid"]

for i, kernel in enumerate([PUK_kernel,"rbf","sigmoid"]):
    result = []
    for j, C in enumerate(np.logspace(-10, 10, num=20)):
        # 分为4类的表现
        result += CV(X_masked, Y, SVC, {
            "kernel": kernel,
            "class_weight": "balanced",
            "C": C,
            "gamma": "scale",
        },
            numlabel=True)
        # 分为2类的表现
        result += CV(X_masked, numout2boolout(Y), SVC, {
            "kernel": kernel,
            "class_weight": "balanced",
            "C": C,
            "gamma": "scale",
        },
            numlabel=False
        )
    np.save("./out/SVC_{}.npy".format(kernelName[i]))
