import numpy as np
import onnxruntime as rt
from sklearn import metrics

sess = rt.InferenceSession("data/rf_iris.onnx", providers=["CPUExecutionProvider"])

X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",")

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

y = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
score = metrics.accuracy_score(y_test, y)
print(score)
