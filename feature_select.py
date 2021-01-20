'''
created: 2020/1/2
'''


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, confusion_matrix


from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import plot_tree

from utils import CV_author, PUK_kernel, numout2boolout, reinterpret


Xx = np.load("./processedData/X/Xx.npy")
Y = np.load("./processedData/Y/Y_train.npy")
y = np.load("./processedData/Y/y_test.npy")

X_feature_name = np.load("./processedData/X/X_featureName.npy")
rM = ['XXXinorg1', "XXXinorg2", "XXXinorg3", "XXXorg1", "XXXorg2"]
Mask = [i for i in range(len(X_feature_name)) if X_feature_name[i] not in rM]
Xx = Xx[:, Mask]
X = Xx[:len(Y), :]
x = Xx[len(Y):, :]
X_feature_name = X_feature_name[Mask]

# 使用SelectKBest方法进行特征选择()


def use_SelectKBest(n, print_log=True, interpret_tree=False, cv_author=False):
    """in:
    n: int, # of features to be selected
    print_log: bool
    out:
    precision: float, precision_score of svm using choosed features
    selected: array(bool array), shape(n_features), True for features selected else False
    """
    skb = SelectKBest(k=n)
    X_trans = skb.fit_transform(X, Y)
    selected = skb.get_support()
    if print_log:
        print('features selected by SelectKBest:')
        print(X_feature_name[selected])

    model = SVC(kernel=PUK_kernel, class_weight="balanced", C=1)
    model.fit(X_trans, Y)
    pred = model.predict(skb.transform(x))
    precision = precision_score(numout2boolout(y), numout2boolout(pred))
    cm = confusion_matrix(numout2boolout(y), numout2boolout(pred))
    if print_log:
        print('precision={0:.3f}'.format(precision))
        print("confusion matrix:")
        print(cm)
    if cv_author:
        CV_author(X_trans, Y, 3, SVC, {
                  "kernel": PUK_kernel, "class_weight": "balanced", "C": 1})

    if interpret_tree:
        tree = reinterpret(X_trans, model, X_trans)
        plt.figure(dpi=160, figsize=(24, 5))
        plot_tree(tree, max_depth=5, feature_names=X_feature_name[selected],
                  rounded=True, filled=True, fontsize=5)
        plt.savefig("./decision_tree_skb.jpg")

    return precision, selected
# features selected by SelectKBest:
# ['temp' 'slowCool' 'pH' 'Na' 'K' 'PaulingElectronegMean'
#  'PaulingElectronegGeom' 'purity']
# precision=0.821

# 确定合适的特征保留个数


def decide_num():
    precisions = []
    for n in range(1, 10):
        precisions.append(use_SelectKBest(n, print_log=False)[0])
    plt.plot(np.arange(1, 10), precisions, marker='.')
    plt.xlabel("# features")
    plt.ylabel("precision")
    plt.savefig("./precision_skb.jpg")

# 使用RFE(recursive feature elimination)方法进行特征选择
def use_RFE(n, print_log=True, interpret_tree=False):
    """in:
    n: int, # of features to be selected
    print_log: bool
    out:
    precision: float, precision_score of svm using choosed features
    selected: array(bool array), shape(n_features), True for features selected else False
    """
    rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=n)
    X_trans = rfe.fit_transform(X, Y)
    selected = rfe.get_support()
    if print_log:
        print('features choosed by RFE:')
        print(X_feature_name[selected])

    model = SVC(kernel=PUK_kernel, class_weight="balanced", C=1)
    model.fit(X_trans, Y)
    pred = model.predict(rfe.transform(x))
    precision = precision_score(numout2boolout(y), numout2boolout(pred))
    cm = confusion_matrix(numout2boolout(y), numout2boolout(pred))
    if print_log:
        print('precision={0:.3f}'.format(precision))
        print("confusion matrix:")
        print(cm)

    if interpret_tree:
        tree = reinterpret(X_trans, model, X_trans)
        plt.figure(dpi=160, figsize=(24, 5))
        plot_tree(tree, max_depth=5, feature_names=X_feature_name[selected],
                  rounded=True, filled=True, fontsize=5)
        plt.show()
        plt.savefig("./decision_tree_rfe.jpg")

    return precision, selected
# features choosed by RFE:
# ['XXXinorg2mass' 'XXXoxlike1mass' 'time' 'pH' 'orgASA_HGeomAvg'
#  'PaulingElectronegGeom' 'hardnessMinWeighted' 'purity']
# precision=0.818
# confusion matrix:
# [[  1  50]
#  [  1 224]]


if __name__ == "__main__":
    # use_SelectKBest(n=8, interpret_tree=True)
    # decide_num()
    use_RFE(n=8, interpret_tree=False)
