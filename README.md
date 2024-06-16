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


