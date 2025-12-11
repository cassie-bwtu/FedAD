import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", dataset_info="", flops_level="", model_name="", goal="", num_clients=100, static=True, ratio=1.0, times=1):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, dataset_info, flops_level, model_name, goal, num_clients, static, ratio, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))

def get_all_results_for_one_algo(algorithm="", dataset="", dataset_info="", flops_level="", model_name="", goal="", num_clients=100, static=True, ratio=1.0, times=1):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = algorithms_list[i] + "_" + model_name + '_' + str(num_clients) + '_static' + str(static) + "_ratio" + str(ratio) + "_" + goal + "_" + str(i)
        # file_name = dataset + "_" + dataset_info + "_" + algorithms_list[i] + "_" + model_name + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(dataset, dataset_info, flops_level, file_name, delete=False)))

    return test_acc

def read_data_then_delete(dataset, dataset_info, flops_level, file_name, delete=False):
    path = os.path.join("../results", dataset, dataset_info, flops_level)
    file_path = os.path.join(path, file_name + ".h5")

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc
