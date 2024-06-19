import numpy as np
import joblib
from sklearn import metrics
import time

lTime = time.perf_counter()
clr = joblib.load("data/rf_iris.joblib")

X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",")

sTime = time.perf_counter()
y = clr.predict(X_test)
score = metrics.accuracy_score(y_test, y)
eTime = time.perf_counter()
print("Accuracy: ", score)
total_duration = (eTime - lTime) * 1000
infer_duration = (eTime - sTime) * 1000
print(f'Total run time: {total_duration:.3f}ms')
print(f'Inference time: {infer_duration:.3f}ms')
