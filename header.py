import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.tree import plot_tree
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


Xx = np.load("./processedData/Xx.npy")
Y = np.load("./processedData/Y.npy")
y = np.load("./processedData/y.npy")
X = Xx[:len(Y), :]
x = Xx[len(Y):, :]
reactantCombination = np.load(
    "./processedData/reactComb.npy", allow_pickle=True)


def numout2boolout(label):  # 结果有1，2，3，4。但是3，4对应人的预测1，1，2
    return label > 2.5


def PUK_kernel(X1, X2, sigma=1.0, omega=1.0):  # 作者使用的SVM核
    # Compute the kernel matrix between two arrays using the Pearson VII function-based universal kernel.

    # Compute squared euclidean distance between each row element pair of the two matrices
    if X1 is X2:
        kernel = squareform(pdist(X1, 'sqeuclidean'))
    else:
        kernel = cdist(X1, X2, 'sqeuclidean')

    kernel = (1 + (kernel * 4 * np.sqrt(2**(1.0/omega)-1)) / sigma**2) ** omega
    kernel = 1/kernel

    return kernel


def CV_author(X, Y, n_splits, Model, params, shuffle=True):
    X_std = StandardScaler().fit_transform(X)  # 有数据没有归一化的风险
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    print(params)
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
        Y_test = numout2boolout(Y_test)
        pred = numout2boolout(pred)
        print("recall={:.3f}".format(
            recall_score(Y_test, pred, average='weighted')))
        print("precision={:.3f}".format(
            precision_score(Y_test, pred, average="weighted")))
        print("accuracy={:.3f}".format(accuracy_score(Y_test, pred)))
        print("confusion matrix is")
        print(confusion_matrix(Y_test, pred))
