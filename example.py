import header

# cross validation
header.CV_author(header.X, header.Y, 3, header.SVC, {
                 "kernel": header.PUK_kernel,
                 "class_weight": "balanced", "C": 1})
# use test set to predict
svc = header.SVC(kernel=header.PUK_kernel, class_weight="balanced", C=1)
svc.fit(header.X, header.Y)
pred = svc.predict(header.x)
print(header.precision_score(header.numout2boolout(
    header.y), header.numout2boolout(pred)))
