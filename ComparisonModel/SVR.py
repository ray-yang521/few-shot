import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR

from utils.evaluation import evaluation_model

train_div_all = 0.45

excel_path = ""
df = pd.read_excel(excel_path)
data = df.values
x = data[:, 0:-2]
y = data[:, -1]
mark = int(len(y) * train_div_all)
x_train = x[:mark]
y_train = y[:mark]
x_test = x[mark:]
y_test = y[mark:]

svr_rbf = SVR(kernel='rbf', C=10, gamma=0.00001)
svr_linear = SVR(kernel='linear', gamma=0.00001, C=2)
svr_poly = SVR(kernel='poly', C=10, degree=1, gamma='auto')

svr_linear.fit(x_train, y_train)
svr_rbf.fit(x_train, y_train)
svr_poly.fit(x_train, y_train)

y_hat1 = svr_linear.predict(x_test)
y_hat2 = svr_rbf.predict(x_test)
y_hat3 = svr_poly.predict(x_test)

nse1, mae1, rmse1, re1, aic1, bic1 = evaluation_model(y_hat1, y_test, 9)
nse2, mae2, rmse2, re2, aic2, bic2 = evaluation_model(y_hat2, y_test, 9)
nse3, mae3, rmse3, re3, aic3, bic3 = evaluation_model(y_hat3, y_test, 9)

print("linear得分:", nse1, mae1, rmse1, re1, aic1, bic1)
print("rbf得分:", nse2, mae2, rmse2, re2, aic2, bic2)
print("poly得分:", nse3, mae3, rmse3, re3, aic3, bic3)

r = len(x_test) + 1
plt.title('SVR')
plt.plot(np.arange(1, r), y_test, 'g.-', label="real")
plt.plot(np.arange(1, r), y_hat1, 'ro-.', label="linear_predict")
plt.plot(np.arange(1, r), y_hat2, 'b.-', label="rbf_predict")
plt.plot(np.arange(1, r), y_hat3, 'y.-', label="poly_predict")
plt.legend()
plt.show()
