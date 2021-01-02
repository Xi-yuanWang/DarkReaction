import header
import numpy as np
# cross validation
#header.CV_author(header.X, header.Y, 3, header.SVC, {
#                 "kernel": header.PUK_kernel,
#                 "class_weight": "balanced", "C": 1})
# use test set to predict
#svc = header.SVC(kernel=header.PUK_kernel, class_weight="balanced", C=1)
#svc.fit(header.X, header.Y)
y=np.load("./processedData/y.npy")
np.save("./processedData/y_test.npy",y)
np.save("./processedData/Y_train.npy",header.Y)
print(y.shape)
pred = svc.predict(header.x)
print(header.precision_score(header.numout2boolout(
    y), header.numout2boolout(pred)))
