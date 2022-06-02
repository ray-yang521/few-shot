import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from utils.evaluation import evaluation_model

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

clf = MLPRegressor(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(10, 10), random_state=1)
clf.fit(feature_train, target_train)
predict_results = clf.predict(feature_test)

nse, mae, rmse, re, aic, bic = evaluation_model(predict_results, target_test, 9)

print("mlp得分:", r2_score(predict_results, target_test), rmse, mae, nse)
print(nse, mae, rmse, re, aic, bic)

# plot figure
plt.title('mlp')
r = len(feature_test) + 1
plt.plot(np.arange(1, r), predict_results, 'ro-', label="mlp_predict")
plt.plot(np.arange(1, r), y[mark:], 'g.-', label="real")
x_min, x_max, y_min, y_max = plt.axis()
x_range = x_max - x_min
y_range = y_max - y_min
plt.text(x_min + 0.02 * x_range, y_max - 0.3 * y_range,
         "R2      = %.4f\nRMSE = %.4f\nMAE   = %.4f,\nNSE    = %.4f" % (
             r2_score(predict_results, target_test), rmse, mae, nse))
plt.legend()
plt.savefig("")
plt.show()
