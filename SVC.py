#!/nfs-share/home/1800011848/miniconda3/envs/torch_geo/bin/python
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH -c 18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1800011848@pku.edu.cn
#SBATCH -J test
#SBATCH -o out
#SBATCH -e error
#SBATCH --time=48:00:00
#SBATCH --qos=normal

from joblib import Parallel, parallel_backend, delayed
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
os.chdir("/nfs-share/home/1800011848/torch_hw/final")

def PUK_kernel(X1, X2, sigma=1.0, omega=1.0):  # 作者使用的SVM核
    if X1 is X2:
        kernel = squareform(pdist(X1, 'sqeuclidean'))
    else:
        kernel = cdist(X1, X2, 'sqeuclidean')

    kernel = (1 + (kernel * 4 * np.sqrt(2**(1.0/omega)-1)) / sigma**2) ** omega
    kernel = 1/kernel

    return kernel
serieLabel = [[], [], [], []]
result = []
series = pd.Series(data=result, index=serieLabel)
np.save("./processedData/SVC_param.npy", series)
Xx = np.load("./processedData/Xx.npy")
Y = np.load("./processedData/Y.npy")
y = np.load("./processedData/y.npy")
X = Xx[:len(Y), :]
X_featureName = np.load("./processedData/X_featureName.npy")
rM = ['XXXinorg1', "XXXinorg2", "XXXinorg3", "XXXorg1", "XXXorg2"]
Mask = [i for i in range(len(X[0])) if X_featureName[i] not in rM]
X_masked = X[:, Mask]

reactantCombination = np.load(
    "./processedData/reactComb.npy", allow_pickle=True)


def numout2boolout(label):  # 结果有1，2，3，4。但是3，4对应人的预测1，1，2
    return label > 2.5


def CV(X, Y, Model, params, n_splits=3, numlabel=True, shuffle=True):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
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


kernelName = ["linear","poly"]  # ''' "PUK", "linear", ''' "poly",rbf", "sigmoid", "precomputed"    "linear", "poly"
'''
with parallel_backend('threading'):
    print(Parallel()(delayed(CV)(X, Y, SVC, {
        "kernel": "rbf",
        "class_weight": "balanced",
        "C": C,
        "gamma": "scale",
    },
        numlabel=True) for C in range(1, 5)))
'''
# header.PUK_kernel, "linear","poly",
for i, kernel in enumerate(["linear","poly"]):
    for j, C in enumerate(np.logspace(-10, 10, num=20)):
        '''
        serieLabel[0] += [kernelName[i]]*6
        serieLabel[1] += ["numY"]*3+["boolY"]*3
        serieLabel[2] += [j]*6
        serieLabel[3] += ["rec", "prec", "acc"]*2
        '''
        result += CV(X, Y, SVC, {
            "kernel": kernel,
            "class_weight": "balanced",
            "C": C,
            "gamma": "scale",
        },
            numlabel=True)
        '''
        result += CV(header.X, header.Y, header.SVC, {
            "kernel": kernel,
            "class_weight": "balanced",
            "C": C,
            "gamma": "scale",
        },
            numlabel=False
        )
        '''
    np.save("./processedData/SVC_{}.npy".format(kernelName[i]), result)
    result.clear()
