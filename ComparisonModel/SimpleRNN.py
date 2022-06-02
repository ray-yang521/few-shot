import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from torch.autograd import Variable

from utils.evaluation import evaluation_model

# Hyper Parameters
EPCOH = 3000
HIDDEN_CELLS = 64
LR = 0.01
LAYER_NUM = 5
INPUT_SIZE = 7
TIME_STEP = 5
OUTPUT_SIZE = 1
train_div_all = 0.2
stop_nse = 0.89
continue_times = 5


def plot_image(the_rmse, the_mae, the_nse, the_loss, pred, var_y, name):
    print("RMSE = %.4f, MAE = %.4f, NSE = %.4f, loss = %.4f\n" % (the_rmse, the_mae, the_nse, the_loss))
    plt.title(name)
    plt.plot(pred, 'r', label='prediction', marker='x')
    plt.plot(var_y, 'b', label='real', marker='.')
    x_min, x_max, y_min, y_max = plt.axis()
    x_range = x_max - x_min
    y_range = y_max - y_min

    if the_nse <= -1:
        plt.text(x_min + 0.02 * x_range, y_min + 0.97 * y_range,
                 "RMSE = %.4f\nMAE = %.4f\nNSE = %.1f,\nLoss = %.4f" % (the_rmse, the_mae, the_nse,
                                                                        the_loss),
                 horizontalalignment='left',
                 verticalalignment='top', family="DejaVu Sans", color="r", style="italic", weight="light",
                 bbox=dict(facecolor="black", alpha=0.2))
    else:
        plt.text(x_min + 0.02 * x_range, y_min + 0.97 * y_range,
                 "RMSE = %.4f\nMAE = %.4f\nNSE = %.4f,\nLoss = %.4f" % (the_rmse, the_mae, the_nse,
                                                                        the_loss),
                 horizontalalignment='left',
                 verticalalignment='top', family="DejaVu Sans", color="r", style="italic", weight="light",
                 bbox=dict(facecolor="black", alpha=0.2))
    plt.legend(loc='upper right')
    plt.show()


df1 = pd.read_excel("")

datas = df1.values

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")

max_value = np.max(datas)
min_value = np.min(datas)
scalar = max_value - min_value
datas = list(map(lambda x: x / scalar, datas))
datas = np.asarray(datas)


def creat_dataset(dataset, look_back):
    data_x = []
    data_y = []
    for i in range(len(dataset) - look_back):
        data_x.append(dataset[i:i + look_back])
        data_y.append(dataset[i + look_back, -1])
    return np.asarray(data_x), np.asarray(data_y)


dataX, dataY = creat_dataset(datas, TIME_STEP)

train_size = int(len(dataX) * train_div_all)

x_train = dataX[:train_size]
y_train = dataY[:train_size]

x_train = x_train.reshape(-1, TIME_STEP, INPUT_SIZE)
y_train = y_train.reshape(-1, 1, 1)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.lstm = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_CELLS,
            num_layers=LAYER_NUM,
            batch_first=True,
        )
        self.Linear1 = nn.Linear(TIME_STEP * HIDDEN_CELLS, TIME_STEP * HIDDEN_CELLS)
        self.Linear2 = nn.Linear(TIME_STEP * HIDDEN_CELLS, TIME_STEP * HIDDEN_CELLS)
        self.Linear_N21 = nn.Linear(TIME_STEP * HIDDEN_CELLS, 1)

    def forward(self, x):
        x1, _ = self.lstm(x)
        batch, time_step, hidden_size = x1.shape
        Linear_out1 = self.Linear1(x1.reshape(batch, -1))
        Linear_out2 = self.Linear1(Linear_out1)
        Linear_out = self.Linear_N21(Linear_out2)
        out = Linear_out.view(batch, -1, 1)
        return out


rnn = RNN()

rnn = rnn.to(device)

x_train = x_train.to(device)
y_train = y_train.to(device)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

count = 0
for each in range(EPCOH):
    var_x = Variable(x_train).type(torch.cuda.FloatTensor)
    var_y = Variable(y_train).type(torch.cuda.FloatTensor)
    out = rnn(var_x)
    batch, time_step, hidden_size = np.shape(out)
    out = out.reshape(batch, -1)
    var_y = var_y.reshape(batch, -1)
    loss = loss_func(out, var_y)
    print('Epoch:{}, Loss:{:.5f}'.format(each + 1, loss.item()))
    the_nse, the_mae, the_rmse, re, AIC, BIC = evaluation_model(out.cpu().detach().numpy(),
                                                                var_y.cpu().detach().numpy(),
                                                                INPUT_SIZE)
    print("RMSE = %.4f, MAE = %.4f, NSE = %.4f\n" % (the_rmse, the_mae, the_nse))
    if the_nse >= stop_nse:
        count += 1
    else:
        count = 0
    if count >= continue_times:
        print('连续%d次，NSE>=%f，提前结束训练！' % (continue_times, stop_nse))
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('training finished!')
test_dataX = dataX[train_size:]
dataX1 = test_dataX.reshape(-1, TIME_STEP, INPUT_SIZE)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor)
rnn_cpu = rnn.to(device2)
pred = rnn_cpu(var_dataX)
batch, time_step, hidden_size = np.shape(pred)
loss = loss_func(pred.reshape(batch, -1).float(),
                 torch.from_numpy(dataY[train_size:].reshape(-1, 1, 1)).reshape(batch, -1).float())
print('Loss:', loss.item())
pred_test = pred.view(-1).data.numpy()

pred_real_test = pred_test * scalar
real_dataY = dataY * scalar
test_dataY = real_dataY[train_size:]

mse = mean_squared_error(test_dataY, pred_real_test)
rmse = math.sqrt(mse)
mae = mean_absolute_error(test_dataY, pred_real_test)
average = 0
fc = 0
s = 0
s = sum(pred_real_test)
average = s / len(pred_real_test)
for each in pred_real_test:
    fc += (each - average) * (each - average)
fc = fc / len(pred_real_test)
nse = 1 - (mse / fc)
print("RMSE = %.4f, MAE = %.4f, NSE = %.4f\n" % (rmse, mae, nse))
plot_image(rmse, mae, nse, loss.item(), pred_real_test, test_dataY, 'LSTM_test')
print("得分:", r2_score(test_dataY, pred_real_test))
print(evaluation_model(pred_real_test, test_dataY, INPUT_SIZE))
print("Testing finished!")
