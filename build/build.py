# Train a model.
import numpy as np
from sklearn import datasets, model_selection, ensemble

import joblib
from skl2onnx import to_onnx

iris = datasets.load_iris()
X, y = iris.data, iris.target
features, targets = iris.feature_names, iris.target_names

X = X.astype(np.float32)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
clr = ensemble.RandomForestClassifier()
clr.fit(X_train, y_train)

# Write to joblib file for inference in python
joblib.dump(clr, "data/rf_iris.joblib")

# Write to onnx file for inference in python and go
onx = to_onnx(clr, X[:1])
with open("data/rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

