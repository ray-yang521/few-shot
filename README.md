# The Fusion Model
Source code for [A new method for runoff prediction in data-scarce region: few-shot learning approach] runnable on GPU and CPU.

Please check out [A new method for runoff prediction in data-scarce region: few-shot learning approach] for PyTorch Implementation (and more). 

## Instructions
Here are the instructions to use the code base

### Dependencies
This code is written in python. To use it you will need:
* [Anaconda](https://www.continuum.io/) - Anaconda includes all the Python-related dependencies

### Prepare Data
Download datasets to the 'data' folder from China Meteorological Data Service Centre

### Train Models
To train the model, try the command line below for detailed guide:
```
python train_models.py
```

### Test Models
To evaluate (dev or test) and save results, use the command line below for detailed guide:
```
python test_models.py
```

### Comparison Models
To compare with other data-driven models, try the command line:
```
cd ComparisonModel
```
and then try the following command line:
```
python ANN.py
python ARMA.py
python BiLSTM.py
python GRU.py
python LSTM.py
python RF.py
python SimpleRNN.py
python SVR.py
```
# few-shot
# few-shot
# few-shot
