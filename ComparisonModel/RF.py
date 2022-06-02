import warnings

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

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

warnings.filterwarnings("ignore")

rf1 = DecisionTreeRegressor()
rf2 = RandomForestRegressor(n_estimators=1000)
rf3 = ExtraTreesRegressor()

y_rf1 = rf1.fit(feature_train, target_train).predict(feature_test)
y_rf2 = rf2.fit(feature_train, target_train).predict(feature_test)
y_rf3 = rf3.fit(feature_train, target_train).predict(feature_test)

print(evaluation_model(y_rf1, target_test, 9))
print(evaluation_model(y_rf2, target_test, 9))
print(evaluation_model(y_rf3, target_test, 9))
