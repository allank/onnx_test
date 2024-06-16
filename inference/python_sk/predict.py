import numpy as np
import joblib
from sklearn import metrics

clr = joblib.load("data/rf_iris.joblib")

X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",")

y = clr.predict(X_test)

score = metrics.accuracy_score(y_test, y)
print(score)
