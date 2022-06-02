import numpy as np
import pandas as pd
import torch

"""
DataManager implements:
read excel data
read specific columns of excel
load data to gpu
create dataset by the specific input dimension and return x_train, y_train
"""


class DataManager:

    def __init__(self, data_dir, time_step, epoch=800, hidden_cells=64, learning_rate=0.02, layer_num=10,
                 input_size=23, output_size=1, train_div_all=0.7):
        self.data_dir = data_dir
        self.epoch = epoch
        self.hidden_cells = hidden_cells
        self.learning_rate = learning_rate
        self.layer_num = layer_num
        self.input_size = input_size
        self.time_step = time_step
        self.output_size = output_size
        self.train_div_all = train_div_all

    def read_excel_data(self):
        """

        @return:
        """
        df = pd.read_excel(self.data_dir)
        df = df[50:400]
        datas = df.values

        max_value = np.max(datas)
        min_value = np.min(datas)
        scalar = max_value - min_value
        return datas, scalar

    def read_specific_column(self, col=-1):
        """

        @param col:
        @return:
        """
        df = pd.read_excel(self.data_dir)
        data = df.iloc[:, col]
        return data

    @staticmethod
    def load_data_to_gpu(gpu_num=0):
        cuda = 'cuda:' + str(gpu_num)
        device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        device2 = torch.device("cpu")
        return device, device2

    def creat_dataset(self, dataset):
        data_x = []
        data_y = []
        look_back = self.time_step
        for i in range(len(dataset) - look_back):
            data_x.append(dataset[i:i + look_back])
            data_y.append(dataset[i + look_back, -1])
        return np.asarray(data_x), np.asarray(data_y)
