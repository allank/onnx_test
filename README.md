# ONNX Test in Go

## Stes to reproduce

### Setup Python environment

```
conda create -n "onnx" python=3.12
conda activate onnx
conda install conda-forge::scikit-learn
conda install conda-forge::skl2onnx
conda install conda-forge::onnxruntime
```

### Build models

`python build/build.py`

This will create:

 - `data/rf_iris.joblib` the model persisted using Python `joblib`
 - `data/rf_iris.onnx` the model persisted using ONNX `skl2onnx`
 - `data/X_test.csv` the test data
 - `data/y_test.csv` the test ground truth values

## Test models

Test using the model loaded from `joblib`:

```
python inference/python_sk/predict.py
0.9473684210526315
```

Test using the model loaded from `onnxruntime`:

```
python inference/python_onnx/predict.py
0.9473684210526315
```



