'''
created: 2020/1/2
'''


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score


from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import plot_tree

from header import CV_author, PUK_kernel, numout2boolout, reinterpret



Xx = np.load("./processedData/Xx.npy")
Y = np.load("./processedData/Y_train.npy")
y = np.load("./processedData/y_test.npy")
X = Xx[:len(Y), :]
x = Xx[len(Y):, :]

X_feature_name = np.load("./processedData/X_featureName.npy")

# 使用SelectKBest方法进行特征选择()
def use_SelectKBest(n, print_log=True, interpret_tree=False):
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

    model=SVC(kernel=PUK_kernel,class_weight="balanced",C=1)
    model.fit(X_trans,Y)
    pred=model.predict(skb.transform(x))
    precision = precision_score(numout2boolout(y),numout2boolout(pred))
    if print_log:
        print('precision={0:.3f}'.format(precision))

    if interpret_tree:
        tree = reinterpret(X_trans, model, X_trans)
        plot_tree(tree, max_depth=5, feature_names=X_feature_name[selected])
        plt.savefig("./decision_tree_skb.jpg")
    
    return precision, selected
# features selected by SelectKBest:     
# ['XXXinorg1' 'XXXinorg2' 'temp' 'slowCool' 'PaulingElectronegMean'
#  'PaulingElectronegGeom' 'purity']    
# precision=0.865

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

    model=SVC(kernel=PUK_kernel,class_weight="balanced",C=1)
    model.fit(X_trans,Y)
    pred=model.predict(rfe.transform(x))
    precision = precision_score(numout2boolout(y),numout2boolout(pred))
    if print_log:
        print('precision={0:.3f}'.format(precision))

    if interpret_tree:
        tree = reinterpret(X_trans, model, X_trans)
        plot_tree(tree, max_depth=5, feature_names=X_feature_name[selected])
        plt.savefig("./decision_tree_skb.jpg")
    
    return precision, selected
# features choosed by RFE:
# ['XXXinorg2mass' 'XXXoxlike1mass' 'time' 'orgASA_HGeomAvg'
#  'PaulingElectronegMean' 'hardnessMinWeighted' 'purity']
# precision= 0.815

if __name__ == "__main__":
    use_SelectKBest(n=7, interpret_tree=True)