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
通过交叉验证方法检验不同正则化强度的线性回归方法的效果
将输出重定向到了./out/Ridge.out, ./out/Lasso.out, ./out/Logistic.out
'''
import sys
import numpy as np
from utils import CV_author, Y, y, numout2boolout
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

# 简单的线性回归
X_masked = np.load("./processedData/X/X_train_masked.npy")
CV_author(X_masked, Y, 3, LinearRegression, {})

# 使用降维后的数据线性回归, 性能差于不降维线性回归
X_reduced=np.load("./processedData/X/X_train_reduced.npy")
CV_author(X_reduced, Y, 3, LinearRegression, {})

# 岭回归
with open("./out/Ridge.out") as f:
    sys.stdout = f
    for alpha in np.logspace(-5, 5):
        CV_author(X_masked, Y, 3, Ridge, {"alpha": alpha})

# Lasso回归
with open("./out/Lasso.out") as f:
    sys.stdout = f
    for alpha in np.logspace(-5, 5):
        CV_author(X_masked, Y, 3, Lasso, {"alpha": alpha})

# 逻辑回归
with open("./out/Logistic.out") as f:
    sys.stdout = f
    for alpha in np.logspace(-5, 5):
        CV_author(X_masked, Y, 3, LogisticRegression, {"C": 1/alpha})
