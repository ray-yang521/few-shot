from itertools import product
from warnings import simplefilter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARMA

from utils.evaluation import evaluation_model

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

# load data
train_div_all = 0.45

excel_path = ""
df = pd.read_excel(excel_path)
data = df.values
scaler = StandardScaler()
scaler.fit(data)
x = data[:, 0:-2]
y = data[:, -1]
mark = int(len(y) * train_div_all)
feature_train = x[:mark]
target_train = y[:mark]
feature_test = x[mark:]
target_test = y[mark:]

# training model
ps = range(0, 6)
qs = range(0, 5)
parameters = product(ps, qs)
parameters_list = list(parameters)

best_aic = float('inf')
results = []
for param in parameters_list:
    try:
        model = ARMA(target_train, order=(param[0], param[1])).fit()
    except ValueError:
        print("参数错误：", param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = model.aic
        best_param = param
    results.append([param, model.aic])
results_table = pd.DataFrame(results)
results_table.columns = ['parameters', 'aic']
print("最优模型", best_model.summary())

# model testing and evaluating
predict_results = best_model.predict(start=mark, end=len(y) - 1)
nse, mae, rmse, re, aic, bic = evaluation_model(predict_results, target_test, 9)
print("mlp得分:", r2_score(predict_results, target_test), rmse, mae, nse)
print(nse, mae, rmse, re, aic, bic)
plt.title('ARMA')

r = len(feature_test) + 1
plt.plot(np.arange(1, r), predict_results, 'r.-', label="ARMA_predict")
plt.plot(np.arange(1, r), y[mark:], 'g-', label="real")
x_min, x_max, y_min, y_max = plt.axis()
x_range = x_max - x_min
y_range = y_max - y_min
plt.text(x_min + 0.02 * x_range, y_max - 0.3 * y_range,
         "R2      = %.4f\nRMSE = %.4f\nMAE   = %.4f,\nNSE    = %.4f" % (
             r2_score(predict_results, target_test), rmse, mae, nse))
plt.legend()
plt.savefig("")
plt.show()
