import os
import shutil
import sys
import torch
from modules.DataManager import DataManager


def del_folder(the_del_folder):
    os.makedirs(the_del_folder, exist_ok=True)
    files = os.listdir(the_del_folder)
    for f in files:
        filepath = os.path.join(the_del_folder, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
    shutil.rmtree(the_del_folder, True)
    os.makedirs(the_del_folder, exist_ok=True)


def preprocess(excel_path, save_result_image_path,cuda_num, time_step, train_div_all, input_size):
    # =======================================================
    if time_step % 2 == 0:
        print('TIME_STEP must be odd!')
        sys.exit()
    del_folder(save_result_image_path)

    dm = DataManager(excel_path, time_step)
    device_gpu, device_cpu = DataManager.load_data_to_gpu(cuda_num)
    data, scalar = dm.read_excel_data()
    data_x, data_y = dm.creat_dataset(data)
    normal_data = []
    for i in data_y:
        normal_data.append(float('%.2f' % i))
    # =======================================================
    train_size = int(len(data_x) * train_div_all)
    # =======================================================
    x_train = data_x[:train_size]
    y_train = data_y[:train_size]

    x_test = data_x[train_size:]
    y_test = data_y[train_size:]
    # =======================================================
    x_train = x_train.reshape(-1, time_step, input_size)
    y_train = y_train.reshape(-1, 1, 1)

    x_test = x_test.reshape(-1, time_step, input_size)
    y_test = y_test.reshape(-1, 1, 1)
    # =======================================================

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    return x_train, y_train, x_test, y_test, data, scalar, device_gpu, device_cpu
