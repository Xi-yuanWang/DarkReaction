import header
import pandas as pd
import numpy as np
# CV 修改的CV_arthor，return而不是print结果

serieLabel = [[], [], [], []]
result = []
series = pd.Series(data=result, index=serieLabel)
np.save("./processedData/SVC_param.npy", series)


def CV(X, Y, Model, params, n_splits=3, numlabel=True, shuffle=True):
    scaler = header.StandardScaler()
    X_std = scaler.fit_transform(X)
    kf = header.KFold(n_splits=n_splits, shuffle=shuffle)
    rec = 0
    acc = 0
    prec = 0
    for train_index_rc, test_index_rc in kf.split(header.reactantCombination):
        train_index = [
            i for rc in train_index_rc for i in header.reactantCombination[rc]]
        test_index = [
            i for rc in test_index_rc for i in header.reactantCombination[rc]]
        X_train, X_test = X_std[train_index], X_std[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model = Model(**params)
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        if numlabel:
            Y_test = header.numout2boolout(Y_test)
            pred = header.numout2boolout(pred)
        rec += header.recall_score(Y_test, pred, average='weighted')
        prec += header.precision_score(Y_test, pred, average="weighted")
        acc += header.accuracy_score(Y_test, pred)
    return [rec/n_splits, prec/n_splits, acc/n_splits]


kernelName = ["PUK", "linear", "poly", "rbf", "sigmoid", "precomputed"]
for i, kernel in enumerate([header.PUK_kernel, "linear", "poly", "rbf", "sigmoid", "precomputed"]):
    for j, C in enumerate(np.logspace(-10, 10, num=20)):
        serieLabel[0] += [kernelName[i]]*6
        serieLabel[1] += ["numY"]*3+["boolY"]*3
        serieLabel[2] += [j]*6
        serieLabel[3] += ["rec", "prec", "acc"]*2
        result += CV(header.X, header.Y, header.SVC, {
            "kernel": kernel,
            "class_weight": "balanced",
            "C": C,
            "gamma": "scale",
        },
            numlabel=True)
        result += CV(header.X, header.Y, header.SVC, {
            "kernel": kernel,
            "class_weight": "balanced",
            "C": C,
            "gamma": "scale",
        },
            numlabel=False
        )
    series = pd.Series(data=result, index=serieLabel)
    np.save("./processedData/SVC_{}.npy".format(kernelName[i]), series)
    result.clear()
    for k in range(4):
        serieLabel[k].clear()
