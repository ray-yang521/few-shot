import math

from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluation_model(out, var_y, input_size):
    # =======================================================
    mse = mean_squared_error(out, var_y)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(out, var_y)
    fc = 0
    s = sum(out)
    average = s / len(out)
    for each in out:
        fc += (each - average) * (each - average)
    fc = fc / len(out)
    nse = 1 - (mse / fc)
    sum_var_y = sum(var_y)
    temp = 0
    for i in range(len(out)):
        temp = temp + (out[i] - var_y[i])
    re = temp / sum_var_y

    # suppose RSS correspond to normal distribution
    if len(out) == len(var_y):
        RSS = 0
        for each_index in range(len(out)):
            RSS = RSS + (out[each_index] - var_y[each_index]) * (out[each_index] - var_y[each_index])
        n = len(var_y)
        L = -(n / 2) * math.log(2 * math.pi) - (n / 2) * math.log(RSS / n) - n / 2
        k = input_size
        AIC = 2 * k - 2 * L
        BIC = -2 * L + math.log(n) * k

    return nse, mae, rmse, re, AIC, BIC
