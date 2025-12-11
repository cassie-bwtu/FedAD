import csv
import os
import time

import numpy as np
from flcore.clients.client_drop import clientDrop
from flcore.servers.serverbase import Server
from threading import Thread
import random
import copy
import psutil
import torch
from flcore.trainmodel.models import ReconstructRN18
from utils.data_utils import get_info_test


class FedDrop(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients (两类)
        self.set_slow_clients()
        self.set_clients(clientDrop)  #生成clients

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []  #统计每一轮的时间消耗
        self.model_name = args.model_name
        # self.train_time_list = []
        self.time_gap = args.time_gap


    def train(self):
        eval_times = 1
        for i in range(self.global_rounds + 1):
            # s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models()  #client.model 更新

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                # print("Selected clients: {}".format([c.id for c in self.selected_clients]))
                print("\nEvaluate global model")
                self.evaluate()

            # start_tt = time.time()
            for client in self.selected_clients:
                client.train()   #更新 client.sub_model

            # self.train_time_list.append(time.time()-start_tt)

            self.receive_models()
            self.aggregate_parameters()

            # self.Budget.append(time.time() - s_t)
            # print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            print("Cost calculation")
            train_time = 0
            for client in self.selected_clients:
                train_time += client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds']
            print("Total training time: {}".format(train_time))
            self.Budget.append(train_time)

            self.save_cost()

            if len(self.Budget) > 1 and (np.sum(self.Budget) > self.time_gap * 60 * eval_times) and (np.sum(self.Budget[:-1]) <= self.time_gap * 60 * eval_times):
                print(f"\n======================>Training time: {self.time_gap*eval_times} min")
                print("\nEvaluate global model")
                self.evaluate_timegap()
                eval_times += 1

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))  #最高acc
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))  #平均（每一轮）时间消耗

        #print("\nTraining time list:")
        #print(self.train_time_list)
        # print("\nAverage training time cost per round:")
        # print(np.mean(self.train_time_list))

        self.save_results()
        self.save_global_model()
        # self.save_loss()


    def receive_models(self):
        assert len(self.selected_clients) > 0

        self.active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []  #id列表
        self.drop_infos = []  #freeze_info列表
        self.uploaded_samples = []  #clients的训练样本的数量
        self.uploaded_models = []   #clients的model

        for client in self.active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.drop_infos.append(client.drop_info)
                self.uploaded_samples.append(client.train_samples)
                self.uploaded_models.append(client.sub_model)


    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.global_model)

        if self.model_name == 'resnet18':
            update_blocks = ['base', 'ds3', 'ds5', 'ds7', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']
        elif self.model_name == 'resnet34':
            update_blocks = ['base', 'ds4', 'ds8', 'ds14', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == 'resnet50':
            update_blocks = ['base', 'ds1', 'ds4', 'ds8', 'ds14', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == 'resnet6':
            update_blocks = ['base', 'ds2', 'head']
            block_names = ['block1', 'block2']
        elif self.model_name == 'resnet10':
            update_blocks = ['base', 'ds2', 'ds3', 'ds4', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4']
        elif self.model_name == 'mobilenet_v2':
            update_blocks = ['base', 'block1', 'block2', 'block4', 'block7', 'block11', 'block14', 'block17', 'block18',
                             'head']
            block_names = ['block3', 'block5', 'block6', 'block8', 'block9', 'block10', 'block12', 'block13', 'block15',
                           'block16']
        elif self.model_name == 'regnet':
            update_blocks = ['base', 'ds1', 'ds2', 'ds3', 'ds7', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9',
                           'block10', 'block11', 'block12', 'block13']
        elif self.model_name == 'vit':
            update_blocks = ['base', 'layernorm', 'classifier']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
        elif self.model_name == 'bert':
            update_blocks = ['base', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4']

        # updata all
        weights = copy.deepcopy(self.uploaded_samples)
        tot_samples = sum(weights)
        for i, num in enumerate(weights):
            weights[i] = num / tot_samples
        # for each block
        for block_name in update_blocks:
            glo = getattr(self.global_model.blocks, block_name)   # self.global_model.base
            for name, param in glo.state_dict().items():
                param.data.zero_()
            for w, client_model in zip(weights, self.uploaded_models):
                cli = getattr(client_model.blocks, block_name)  # client.sub_model.blocks.ds3
                for (server_name, server_param), (client_name, client_param) in zip(glo.state_dict().items(), cli.state_dict().items()):
                    if 'num_batches_tracked' in server_name:
                        server_param.data += int(client_param.data.clone() * w)
                    else:
                        server_param.data += client_param.data.clone() * w

        # optional updating
        column_sum = list(map(sum, zip(*self.drop_infos)))
        # 对于每一个block
        for i in range(len(block_names)):
            # 有client更新
            if column_sum[i] < len(self.uploaded_models):
                server_block = getattr(self.global_model.blocks, block_names[i])  # self.global_model.block1

                # 置为零
                for name, param in server_block.state_dict().items():
                    param.data.zero_()

                uploaded_weights = []  # client的权重
                uploaded_blocks = []  # 更新的模型
                # 对于每一个client
                for drop_info, client_model, uploaded_samples in zip(self.drop_infos, self.uploaded_models,
                                                                       self.uploaded_samples):
                    # 更新了当前的block
                    if drop_info[i] == 0:
                        uploaded_weights.append(uploaded_samples)
                        client_block = getattr(client_model.blocks, block_names[i])  # client.epoch_model.blocks.block1
                        uploaded_blocks.append(client_block)

                # 计算权重
                total_num = sum(uploaded_weights)
                for j, num in enumerate(uploaded_weights):
                    uploaded_weights[j] = num / total_num

                # 更新全局模型
                for w, client_model in zip(uploaded_weights, uploaded_blocks):
                    for (server_name, server_param), (client_name, client_param) in zip(server_block.state_dict().items(), client_model.state_dict().items()):
                        if 'num_batches_tracked' in server_name:
                            server_param.data += int(client_param.data.clone() * w)
                        else:
                            server_param.data += client_param.data.clone() * w


    def save_cost(self):
        algo = self.algorithm + "_" + self.model_name + '_' + str(self.num_clients) + '_static' + str(
            self.static) + "_ratio" + str(self.ratio)
        result_path = os.path.join("../results", self.dataset, self.dataset_info)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        file_path = os.path.join(result_path, algo + "_TrainCost_" + self.goal + "_" + str(self.times) + ".csv")
        clients_id = []
        clients_traintime = []
        for client in self.selected_clients:
            clients_id.append(client.id)
            clients_traintime.append(client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'])
        with open(file_path, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(clients_id)
            writer.writerow(clients_traintime)
