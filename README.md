# ONNX Test in Go

## Steps to reproduce

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

### Test models

#### Setting up ONNX Runtime

You will need the ONNX runtime for your platform.  Visit [https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases) and download the release for your platform.  Extract the downloaded files and copy the shared libary to the `inference/go_onnx` folder.  On Intel Mac this was `libonnxruntime.1.18.0.dylib`.  If necessary, update the following line in `main.go`:

`	ort.SetSharedLibraryPath("libonnxruntime.1.18.0.dylib")`

#### Running the tests

Test using the model loaded from `joblib`:

```
python inference/python_sk/predict.py
Accuracy:  0.9736842105263158
Total run time: 150.890ms
Inference time: 4.618ms
```

Test using the model loaded from `onnxruntime`:

First run:

```
python inference/python_onnx/predict.py
Accuracy:  0.9736842105263158
Total run time: 83.337ms
Inference time: 3.119ms5
```
Subsequent runs:

```
python inference/python_onnx/predict.py
Accuracy:  0.9736842105263158
Total run time: 7.824ms
Inference time: 1.015ms
```

Test using the model loaded in Go:

```
./onnx_test
Accuracy:  0.9736842105263158
Total run time:  34.424473ms
Inference time:  395.164µs
```



