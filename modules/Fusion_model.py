import os
from warnings import simplefilter

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.DataManager import DataManager
from utils.evaluation import evaluation_model

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

# hyper parameter
description = 'Lancangjiang'
EPOCHS = 20000
HIDDEN_CELLS = 64
LR = 0.01
LAYER_NUM = 5
INPUT_SIZE = 7
# time step muse be odd
TIME_STEP = 5
OUTPUT_SIZE = 1
train_div_all = 0.45
alpha = 1
beta = 0.01
gamma = 0.1
stop_nse = 0.9
continue_times = 5
classifier = [0.07465577127889449, 0.09399908879555473, 0.17518418195961694, 0.42578298391621605, 0.6176750223039995]
load_model_or_not = True


# =======================================================
def create_hyper_parameters_list(the_classifier):
    hyper_parameters_dict = {'description': description, 'EPOCHS': EPOCHS, 'HIDDEN_CELLS': HIDDEN_CELLS,
                             'LR': LR,
                             'LAYER_NUM': LAYER_NUM,
                             'INPUT_SIZE': INPUT_SIZE, 'TIME_STEP': TIME_STEP, 'OUTPUT_SIZE': OUTPUT_SIZE,
                             'train_div_all': train_div_all, 'alpha': alpha, 'beta': beta, 'gamma': gamma,
                             'stop_nse': stop_nse, 'continue_times': continue_times,
                             'classifier': the_classifier}
    return hyper_parameters_dict


class RNN(nn.Module):

    def __init__(self, hyper_parameters_dict):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=hyper_parameters_dict["INPUT_SIZE"],
            hidden_size=hyper_parameters_dict["HIDDEN_CELLS"],  # rnn hidden unit
            num_layers=hyper_parameters_dict["LAYER_NUM"],  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.Linear1 = nn.Linear(hyper_parameters_dict["TIME_STEP"] * hyper_parameters_dict["HIDDEN_CELLS"],
                                 hyper_parameters_dict["TIME_STEP"] * hyper_parameters_dict[
                                     "HIDDEN_CELLS"])
        self.Linear2 = nn.Linear(hyper_parameters_dict["TIME_STEP"] * hyper_parameters_dict["HIDDEN_CELLS"],
                                 hyper_parameters_dict["TIME_STEP"] * hyper_parameters_dict[
                                     "HIDDEN_CELLS"])
        self.Linear_N21 = nn.Linear(hyper_parameters_dict["TIME_STEP"] * hyper_parameters_dict["HIDDEN_CELLS"],
                                    1)

    def forward(self, x):
        """

        :param x:
        :return:
        """

        x1, (h_n, h_c) = self.lstm(x)
        batch, time_step, hidden_size = x1.shape
        linear_out2 = self.Linear1(x1.reshape(batch, -1))
        linear_out3 = self.Linear2(linear_out2)
        linear_out = self.Linear_N21(linear_out3)
        out = linear_out.view(batch, -1, 1)
        return out, h_n, h_c, x1


class Prototype(object):

    def __init__(self, hyper_parameters_dict, the_aim_data, load_model_dir,
                 load_hyper_parameters_dir,
                 load_result_image_path, save_result_list_path, load_nse_dir, cuda_num):
        self.time_window = hyper_parameters_dict["TIME_STEP"]
        self.classifier = hyper_parameters_dict["classifier"]
        self.aim_data = the_aim_data
        self.load_model_dir = load_model_dir
        self.load_hyper_parameters_dir = load_hyper_parameters_dir
        self.load_result_image_path = load_result_image_path
        self.load_nse_dir = load_nse_dir
        self.save_result_list_path = save_result_list_path
        self.cuda_num = cuda_num
        self.gpu_device = DataManager.load_data_to_gpu(cuda_num)[0]
        self.hyper_parameters_dict = hyper_parameters_dict
        self.model = RNN(hyper_parameters_dict)
        self.representations = {}

    def mapping_step_and_value(self, now_step):
        """
        获取当前时间步的值
        :param now_step: 当前时间步
        :return: 当前时间步的值
        """
        the_data = self.aim_data
        return the_data[now_step]

    def get_step_label(self, now_step):
        """
        给时间步打标签！
        检索当前时间步及其前后各半个时间窗口的value值对应的label，并将value对应的label加入对应时间步的字典中。函数返回时间步的起止点、带label的时间步的字典
        :param now_step: 当前时间步
        :return: start_step ——》 end_step、now_step、每个时间步（key）对应的label（value）（字典输出）
        """
        the_dict = {}
        time_window = self.time_window
        the_classifier = self.classifier
        # =======================================================
        start_step = now_step - time_window // 2
        end_step = now_step + time_window // 2
        # =======================================================
        for each in range(start_step, end_step + 1):
            each_value = self.mapping_step_and_value(each)
            if 0 <= each_value < the_classifier[0]:
                the_dict[each] = 0
            elif each_value >= the_classifier[-1]:
                the_dict[each] = len(the_classifier)
            else:
                for i in range(len(the_classifier) - 1):
                    if the_classifier[i] <= each_value < the_classifier[i + 1]:
                        the_dict[each] = i + 1
        return start_step, end_step, the_dict

    def divide_step_label_to_different_support_sets(self, now_step):
        """
        根据带标签的时间步字典，相同标签的时间步归为一类，初始化的矩阵空白值为-1
        :param now_step: 当前时间步
        for example:
                    the_dict = {0: 2, 1: 3, 2: 4, 3: 5, 4: 4, 5: 3}, the key is time step, the value is label in response.
                    unique_labels = [2 3 4 5]
                    so, after divided:
                    all_support_set = [[ 0. -1. -1. -1. -1. -1.]
                                       [-1.  1. -1. -1. -1.  5.]
                                       [-1. -1.  2. -1.  4. -1.]
                                       [-1. -1. -1.  3. -1. -1.]]
                    -1 means there is no time_step, other values of time_step is in the_dict's keys,
                    every row of all_support_set is a support_set,
                    it's labels can be searched in unique_labels in response.
                    like, all_support_set[0]([ 0. -1. -1. -1. -1. -1.])'s label is unique_labels[0] = 2
        """
        start_step, end_step, the_dict = self.get_step_label(now_step)
        keys = list(the_dict.keys())
        keys_nums = len(keys)

        labels = list(the_dict.values())
        unique_labels = pd.Series(labels).unique()
        unique_labels_nums = len(unique_labels)
        # =======================================================
        # 动态初始化all_support_set的元素为-1
        all_support_set = np.zeros((unique_labels_nums, keys_nums))
        x, y = np.shape(all_support_set)
        for i in range(x):
            for j in range(y):
                all_support_set[i][j] = -1

        for i, each_unique_label in enumerate(unique_labels):
            temp_index_set = []
            for j, each_label in enumerate(labels):
                if each_label == each_unique_label:
                    temp_index_set.append(j)
            for k in temp_index_set:
                all_support_set[i][k] = keys[k]

        return unique_labels, all_support_set

    def save_representations(self, representations):
        batch, time_step, hidden_size = representations.shape

        for i in range(batch):
            self.representations[i + self.time_window] = representations[i]

    def read_representation(self, now_step):
        """

        @rtype: object
        """
        return self.representations[now_step]

    def compute_prototype(self, unique_labels, all_support_set):
        """
        计算给定的一个时间窗口内各个支持集类别的原型。
        :param unique_labels: different support sets' label file.
        :param all_support_set:
        :return:
        """
        rows, cols = all_support_set.shape
        prototype_list = []
        for i in range(rows):
            current_support_set = all_support_set[i]
            same_label_representation = []
            current_step = 0
            for j in range(cols):
                if current_support_set[j] in current_support_set and current_support_set[j] != '-1' \
                        and current_support_set[j] != -1:
                    current_step = current_support_set[j]
                    same_label_representation.append(self.read_representation(current_step))
            temp_representation = torch.zeros(self.read_representation(current_step).shape).to(
                DataManager.load_data_to_gpu(0)[0])
            for each in same_label_representation:
                temp_representation += each
            prototype_list.append(temp_representation)
        return unique_labels, prototype_list

    def compute_prototype_loss(self, now_step, all_support_set, prototype_list):
        """
        计算给定的一个时间窗口内query point属于它原本原型的概率。
        :param now_step:
        :param all_support_set:
        :param prototype_list:
        :return:
        """

        label_index = 0

        query_step_representation = self.read_representation(now_step)  # n个tensor的list
        # search which support set query_step belongs to.
        for i in range(len(all_support_set)):
            for j in range(len(all_support_set[i])):
                if str(all_support_set[i][j]) == str(now_step):
                    label_index = i

        # compute the loss between query step and its prototype.
        # MSE
        temp_loss = torch.nn.MSELoss()(query_step_representation, prototype_list[label_index])
        exp_numerator_loss = torch.exp(-temp_loss)

        # compute the sum loss that query step with all the prototypes.
        denominator_loss_list = []
        denominator_loss = torch.zeros(1, 1)
        for each_prototype in prototype_list:
            # MSE
            temp_loss = torch.nn.MSELoss()(query_step_representation, each_prototype)
            denominator_loss_list.append(torch.exp(-temp_loss))

        for each_loss in denominator_loss_list:
            denominator_loss += each_loss

        return -(torch.log10(exp_numerator_loss / denominator_loss))

    def search_peak_points(self):
        the_aim_data = self.aim_data
        peak_list = []

        for index in range(len(the_aim_data) - 1):
            if the_aim_data[index] >= the_aim_data[index - 1] and the_aim_data[index] >= the_aim_data[index + 1]:
                peak_list.append(index)
        return peak_list

    def search_valley_points(self):
        the_aim_data = self.aim_data
        valley_list = []
        for index in range(len(the_aim_data) - 1):
            if the_aim_data[index] <= the_aim_data[index - 1] and the_aim_data[index] <= the_aim_data[index + 1]:
                valley_list.append(index)
        return valley_list

    @staticmethod
    def calculate_triangle_area(core_point, value_list):
        x1 = core_point - 1
        x2 = core_point
        x3 = core_point + 1
        y1 = value_list[x1]
        y2 = value_list[x2]
        y3 = value_list[x3]
        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        x3 = torch.tensor(x3)

        area = 0.5 * torch.abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)
        return area

    def train(self, the_x_train, the_y_train, device, save_model, save_hyper_parameter, save_nse) -> object:
        # =======================================================
        self.model = self.model.to(device)
        the_x_train = the_x_train.to(device)
        the_y_train = the_y_train.to(device)
        # =======================================================
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hyper_parameters_dict["LR"])
        loss1_mse = nn.MSELoss()
        # =======================================================
        if device.type == 'cuda':
            var_x = Variable(the_x_train).type(torch.cuda.FloatTensor)
            var_y = Variable(the_y_train).type(torch.cuda.FloatTensor)
        elif device.type == 'cpu':
            var_x = Variable(the_x_train).type(torch.FloatTensor)
            var_y = Variable(the_y_train).type(torch.FloatTensor)
        else:
            raise Exception('please input cuda or cpu in response!')
        # =======================================================
        count = 0
        nse_list = []
        best_nse = 0
        for each in range(hyper_parameters_dict["EPOCHS"]):  # all_support_set = torch.from_numpy(all_support_set)

            # scheduler.step()
            out, h_n, h_c, origin_out = self.model(var_x)
            batch, time_step, hidden_size = np.shape(origin_out)

            start_step = self.time_window // 2
            start_step += self.time_window
            end_step = batch + self.time_window // 2
            self.save_representations(origin_out)
            out = out.reshape(batch, -1)
            var_y = var_y.reshape(batch, -1)
            # =======================================================
            # RNN regression loss ---> MSE(pre, var_y)
            loss1 = loss1_mse(out, var_y)

            # =======================================================
            # prototype loss
            loss2 = torch.zeros(1, 1)
            for now_step in range(start_step, end_step):
                each_loss = self.each_loss2(now_step)
                loss2 += each_loss
            loss2 = Variable(loss2).type(torch.cuda.FloatTensor)
            # =======================================================
            # global loss
            loss = self.hyper_parameters_dict["alpha"] * loss1 + self.hyper_parameters_dict["beta"] * loss2
            print('Epoch:{}, Loss:{:.5f}, Loss1:{:.5f}, Loss2:{:.5f}'.format(each + 1, loss.item(),
                                                                             loss1.item(),
                                                                             loss2.item()))
            # =======================================================
            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            the_nse, the_mae, the_rmse, re, AIC, BIC = evaluation_model(out.cpu().detach().numpy(),
                                                                        var_y.cpu().detach().numpy(),
                                                                        self.hyper_parameters_dict["INPUT_SIZE"])
            print('{:.4f}'.format(the_nse[0]))
            if the_nse >= best_nse:
                best_nse = the_nse
            nse_list.append(math.log10(-(the_nse - 1)))
            if the_nse >= hyper_parameters_dict["stop_nse"]:
                count += 1
            else:
                count = 0
            if count >= hyper_parameters_dict["continue_times"]:
                print('连续%d次，NSE>=%f，提前结束训练！' % (
                    hyper_parameters_dict["continue_times"], hyper_parameters_dict["stop_nse"]))
                break
        # =======================================================
        if save_model:
            os.makedirs(self.load_model_dir[:-9], exist_ok=True)
            torch.save(self.model.state_dict(), self.load_model_dir)
        if save_hyper_parameter:
            os.makedirs(self.load_hyper_parameters_dir[:-20], exist_ok=True)
            f = open(self.load_hyper_parameters_dir, 'w')
            f.write(str(self.hyper_parameters_dict))
        if save_nse:
            os.makedirs(self.load_nse_dir[:-12], exist_ok=True)
            torch.save(nse_list, self.load_nse_dir)
        print('best nse is ', best_nse)
        return nse_list

    def each_loss2(self, now_step):
        unique_labels, all_support_set = self.divide_step_label_to_different_support_sets(now_step)
        unique_labels, center_list = self.compute_prototype(unique_labels, all_support_set)
        each_loss = self.compute_prototype_loss(now_step, all_support_set, center_list)
        return each_loss

    def test(self, x_test, y_test, scalar, device, model_dir, load_model=False):
        if load_model:
            self.model.load_state_dict(torch.load(model_dir))
        else:
            self.model = self.model.to(device)
        # =======================================================
        if device.type == 'cpu':
            var_x = Variable(x_test).type(torch.FloatTensor)
            var_y = Variable(y_test).type(torch.FloatTensor)
        else:
            var_x = Variable(x_test).type(torch.cuda.FloatTensor)
            var_y = Variable(y_test).type(torch.cuda.FloatTensor)
        # =======================================================
        pred, h_n, h_c, origin_out = self.model(var_x)
        mse_loss = nn.MSELoss()(pred, y_test)
        print('Loss:', mse_loss.item())

        pred_test = pred.view(-1).data.numpy()
        var_y = var_y.view(-1).data.numpy()

        pred_real_y_test = pred_test * scalar
        real_real_y_test = var_y * scalar

        os.makedirs(self.load_nse_dir[:-12], exist_ok=True)

        torch.save((pred_real_y_test, real_real_y_test), self.save_result_list_path)

        return pred_real_y_test, real_real_y_test, mse_loss


def plot_image(the_rmse, the_mae, the_nse, the_loss, pred, var_y, the_path, name):
    print("RMSE = %.4f, MAE = %.4f, NSE = %.4f, loss = %.4f\n" % (the_rmse, the_mae, the_nse, the_loss))
    plt.title(name)
    plt.plot(pred, 'r', label='prediction', marker='x')
    plt.plot(var_y, 'b', label='real', marker='.')
    x_min, x_max, y_min, y_max = plt.axis()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 根据NSE的位数调整显示小数点后的位数。NSE<=-1，显示小数点后一位；NSE为(-1,1)，显示小数点后四位。
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
    name = name + '.png'
    os.makedirs(the_path, exist_ok=True)
    the_path = os.path.join(the_path, name)
    plt.savefig(the_path)
    plt.show()
