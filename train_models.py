from modules.Fusion_model import *
import time

def train_models():
    # =======================================================
    start_time = time.time()
    excel_path = ""
    model_path = ""
    hyper_parameters_path = ""
    nse_list_path = ""
    save_result_list_path = ""
    save_result_image_path = ""
    cuda_num = 0
    hyper_parameters_dict = create_hyper_parameters_list(classifier)
    # =======================================================
    # 构建数据
    x_train, y_train, x_test, y_test, data, scalar, device_gpu, device_cpu = preprocess(excel_path,
                                                                                        save_result_image_path,
                                                                                        cuda_num,
                                                                                        hyper_parameters_dict[
                                                                                            "TIME_STEP"],
                                                                                        train_div_all,
                                                                                        hyper_parameters_dict[
                                                                                            "INPUT_SIZE"])
    # =======================================================
    # =======================训 练============================
    # =======================================================
    aim_data = data[:, -1]
    prototype = Prototype(hyper_parameters_dict, aim_data, model_path, hyper_parameters_path,
                          save_result_image_path,
                          save_result_list_path, nse_list_path, cuda_num)
    print('=' * 30 + ' 开始训练！' + '=' * 30)
    if not load_model_or_not:
        the_nse_list = prototype.train(x_train, y_train, device_gpu, True, True, True)
    print('=' * 30 + ' 训练结束！' + '=' * 30)
    print('\n')


if __name__ == '__main__':
    train_models()
