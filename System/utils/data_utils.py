import random

import numpy as np
import os
import torch

def read_data(dataset, dataset_info, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, dataset_info, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        return train_data
    else:
        test_data_dir = os.path.join('../dataset', dataset, dataset_info, 'test/')
        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return test_data


def read_client_data(dataset, dataset_info, idx, is_train=True):
    '''
        :param dataset: dataset name
        :param idx: client id
        :param is_train: bool
        :return:
    '''
    if dataset == 'SST2':
        if is_train:
            train_data = read_data(dataset, dataset_info, idx, is_train)
            # 关键修改：对于文本数据使用 LongTensor，保持完整的256维
            X_train = torch.LongTensor(train_data['x'])  # shape: (N, 256)
            y_train = torch.LongTensor(train_data['y'])  # shape: (N,)

            train_data = [(x, y) for x, y in zip(X_train, y_train)]
            return train_data
        else:
            test_data = read_data(dataset, dataset_info, idx, is_train)
            X_test = torch.LongTensor(test_data['x'])  # shape: (N, 256)
            y_test = torch.LongTensor(test_data['y'])  # shape: (N,)

            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            return test_data
    else:
        if is_train:
            train_data = read_data(dataset, dataset_info, idx, is_train)
            X_train = torch.Tensor(train_data['x']).type(torch.float32)  #(70000,1,28,28)
            y_train = torch.Tensor(train_data['y']).type(torch.int64)  #(70000,)

            train_data = [(x, y) for x, y in zip(X_train, y_train)]
            return train_data

        else:
            test_data = read_data(dataset, dataset_info, idx, is_train)
            X_test = torch.Tensor(test_data['x']).type(torch.float32)
            y_test = torch.Tensor(test_data['y']).type(torch.int64)
            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            return test_data



def read_test_data(dataset, dataset_info, ids, is_train=False):
    '''
    :param dataset: dataset name
    :param idx: client id
    :param is_train: bool
    :return:
    '''
    if dataset =='SST2':
        if is_train == False:
            X_list = []
            y_list = []
            for c in ids:
                test_data = read_data(dataset, dataset_info, c, is_train)
                # 关键修改：使用 LongTensor
                X_test = torch.LongTensor(test_data['x'])  # shape: (N, 256)
                y_test = torch.LongTensor(test_data['y'])  # shape: (N,)
                X_list.append(X_test)
                y_list.append(y_test)

            X_test = torch.cat(X_list, dim=0)
            y_test = torch.cat(y_list, dim=0)

            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            return test_data
    else:
        if is_train == False:
            X_list = []
            y_list = []
            for c in ids:
                test_data = read_data(dataset, dataset_info, c, is_train)
                X_test = torch.Tensor(test_data['x']).type(torch.float32)  #(70000,1,28,28)
                y_test = torch.Tensor(test_data['y']).type(torch.int64)  #(70000,)
                X_list.append(X_test)
                y_list.append(y_test)
            X_test = torch.cat(X_list, dim=0)
            y_test = torch.cat(y_list, dim=0)

            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            # print(len(test_data))
            return test_data


def get_info(model_name, id, ratio):
    if model_name == 'resnet18':
        block_num = 8
    elif model_name == 'resnet34':
        block_num = 16
    elif model_name == 'resnet50':
        block_num = 16
    elif model_name == 'regnet':
        block_num = 13
    elif model_name == 'mobilenet_v2':
        block_num = 10
    elif model_name == 'vit':
        block_num = 12
    elif model_name == 'bert':
        block_num = 4
    list = [0 for i in range(block_num)]

    if ratio == 0.3:
        if id >= 30:
            idx = random.sample(range(block_num), (id-30)//10+1)
            for i in idx:
                list[i] = 1
    elif ratio == 0.5:
        if id >= 50:
            idx = random.sample(range(block_num), (id-50)//10+1)
            for i in idx:
                list[i] = 1
    elif ratio == 0.7:
        if id >= 70:
            idx = random.sample(range(block_num), (id-70)//5+1)
            for i in idx:
                list[i] = 1

    return list


def get_info_test(model_name, num):
    if model_name == 'resnet18':
        block_num = 8
    elif model_name == 'resnet34':
        block_num = 16
    elif model_name == 'resnet50':
        block_num = 16
    elif model_name == 'regnet':
        block_num = 13
    elif model_name == 'mobilenet_v2':
        block_num = 10
    elif model_name == 'vit':
        block_num = 12
    elif model_name == 'bert':
        block_num = 4
    list = [0 for i in range(block_num)]
    idx = random.sample(range(block_num), num)
    for i in idx:
        list[i] = 1
    return list


def get_flops(id, rate, flops):
    """
    rate: flops_rate, default='0.5,0.3,0.2',
    flops: flops_level, default='1,0.66,0.33'
    """

    rate_list = rate.split(',')
    flops_level_list = flops.split(',')
    rate_list = [float(r) for r in rate_list]
    flops_level_list = [float(r) for r in flops_level_list]
    cumulative_rate = 0
    for i, r in enumerate(rate_list):
        cumulative_rate += r
        if id < cumulative_rate * 100:
            return flops_level_list[i]



def get_drop_info(model_name, flops_level):
    '''
    flops_level: float (0,1]
    '''
    if model_name == 'resnet18':
        block_num = 8
    elif model_name == 'resnet34':
        block_num = 16
    elif model_name == 'resnet50':
        block_num = 16
    elif model_name == 'resnet6':
        block_num = 2
    elif model_name == 'resnet10':
        block_num = 4
    elif model_name == 'mobilenet_v2':
        block_num = 10
    elif model_name == 'regnet':
        block_num = 13
    elif model_name == 'vit':
        block_num = 12
    elif model_name == 'bert':
        block_num = 4

    policy = torch.ones(block_num, 2)
    policy[:, 1] = 0
    if flops_level != 1:
        drop_num = block_num - int(block_num*flops_level)
        idx = random.sample(range(block_num), drop_num)
        for i in idx:
            policy[i, :] = torch.tensor([0, 1])

    return policy


if __name__=="__main__":
    train_data = read_client_data('SST2', 'noniid_dir', 0, is_train=True)
    print(train_data)
