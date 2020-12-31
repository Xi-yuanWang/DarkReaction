import header

# cross validation
CV_author(X,Y,3,SVC,{"kernel":PUK_kernel,"class_weight":"balanced","C":1})

svc=SVC(kernel=PUK_kernel,class_weight="balanced",C=1)
svc.fit(X,Y)
pred=svc.predict(x)
print(accuracy_score(numout2boolout(y),numout2boolout(pred)))