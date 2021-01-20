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
import numpy as np
from utils import numout2boolout
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 加载数据
x_feature = np.load("./processedData/X/X_featureName.npy")
X = np.load("./processedData/X/X_train.npy")
x = np.load("./processedData/X/x_test.npy")
Y = numout2boolout(np.load("./processedData/Y/Y_train.npy"))
y = numout2boolout(np.load("./processedData/Y/y_test.npy"))

# 创建随机森林并预测
rf0 = RandomForestClassifier(oob_score=True,random_state=10)
rf0.fit(X,Y)
print(rf0.oob_score_)
print("accuracy:%f" % rf0.oob_score_)
rf0.fit(x,y)
print(rf0.oob_score_)
print("accuracy:%f" % rf0.oob_score_)

# 调参
param_test1 = {"n_estimators":range(1,101,5)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_test1,
                        scoring='roc_auc',cv=10)
gsearch1.fit(X,Y)
print(gsearch1.best_params_)
print("best accuracy:%f" % gsearch1.best_score_)
means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:  %r" % (mean,param))

param_test2 = {"max_features":range(1,11,1)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=31,
                        random_state=10),
                        param_grid = param_test2,scoring='roc_auc',cv=10)
gsearch2.fit(X,Y)
print(gsearch2.best_params_)
print("best accuracy:%f" % gsearch2.best_score_)
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:  %r" % (mean,param))

# 使用选择出来的参数给出的最终结果
rf1 = RandomForestClassifier(n_estimators=31,max_features=6,
                             oob_score=True,random_state=10)
rf1.fit(X,Y)
print(rf1.oob_score_)
print("accuracy: %f" % rf1.oob_score_)
rf1.fit(x,y)
print(rf1.oob_score_)
print("accuracy: %f" % rf1.oob_score_)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf1.feature_importances_), x_feature),
             reverse=True))
