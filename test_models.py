from modules.Fusion_model import *
import time

def test_models():
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
    # =======================测 试============================
    # =======================================================
    aim_data = data[:, -1]
    prototype = Prototype(hyper_parameters_dict, aim_data, model_path, hyper_parameters_path,
                          save_result_image_path,
                          save_result_list_path, nse_list_path, cuda_num)
    print('=' * 30 + ' 开始测试！' + '=' * 30)
    pred_real_y_test, real_real_y_test, the_mse_loss = prototype.test(x_test, y_test, scalar, device_cpu, model_path,
                                                                      load_model_or_not)
    nse, mae, rmse, re, AIC, BIC = evaluation_model(pred_real_y_test, real_real_y_test,
                                                    hyper_parameters_dict["INPUT_SIZE"])
    print(nse, mae, rmse, re, AIC, BIC)
    print('AIC=%f, BIC=%f', (AIC, BIC))
    plot_image(rmse, mae, nse, the_mse_loss.item(), pred_real_y_test, real_real_y_test,
               prototype.load_result_image_path, 'test')
    # =======================================================
    print('=' * 30 + ' 测试结束！' + '=' * 30)
    # =======================================================
    end_time = time.time()
    cost_time = end_time - start_time
    print('time cost %s(s)' % str(cost_time))


if __name__ == '__main__':
    test_models()
