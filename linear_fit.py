'''
通过交叉验证方法检验不同正则化强度的线性回归方法的效果
将输出重定向到了./out/Ridge.out, ./out/Lasso.out, ./out/Logistic.out
'''
import sys
import numpy as np
from utils import CV_author, Y, y, numout2boolout
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

X_masked = np.load("./processedData/X/X_masked.npy")

CV_author(X_masked, Y, 3, LinearRegression)
with open("./out/Ridge.out") as f:
    sys.stdout = f
    for alpha in np.logspace(-5, 5):
        CV_author(X_masked, Y, 3, Ridge, {"alpha": alpha})

with open("./out/Lasso.out") as f:
    sys.stdout = f
    for alpha in np.logspace(-5, 5):
        CV_author(X_masked, Y, 3, Lasso, {"alpha": alpha})

with open("./out/Logistic.out") as f:
    sys.stdout = f
    for alpha in np.logspace(-5, 5):
        CV_author(X_masked, Y, 3, LogisticRegression, {"C": 1/alpha})
