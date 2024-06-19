import numpy as np
import onnxruntime as rt
from sklearn import metrics
import time

lTime = time.perf_counter()



sess = rt.InferenceSession("data/rf_iris.onnx", providers=["CPUExecutionProvider"])

X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",")

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

sTime = time.perf_counter()
y = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
score = metrics.accuracy_score(y_test, y)
eTime = time.perf_counter()

print("Accuracy: ", score)
total_duration = (eTime - lTime) * 1000
infer_duration = (eTime - sTime) * 1000
print(f'Total run time: {total_duration:.3f}ms')
print(f'Inference time: {infer_duration:.3f}ms')

