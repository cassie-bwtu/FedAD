import time

import numpy as np
from flcore.clients.client_freeze import clientFreeze
from flcore.servers.serverbase import Server
from threading import Thread
import random
import copy

class FedFreeze(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients (两类)
        self.set_slow_clients()
        self.set_clients(clientFreeze)  #生成clients

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []  #统计每一轮的时间消耗
        self.model_name = args.model_name

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                # print("Selected clients: {}".format([c.id for c in self.selected_clients]))
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))  #最高acc
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))  #平均（每一轮）时间消耗

        self.save_results()
        self.save_global_model()


    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []  #id列表
        self.freeze_infos = []  #freeze_info列表
        self.uploaded_samples = []  #clients的训练样本的数量
        self.uploaded_models = []   #clients的model

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.freeze_infos.append(client.freeze_info)
                self.uploaded_samples.append(client.train_samples)
                self.uploaded_models.append(client.model)


    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        #self.global_model = copy.deepcopy(self.global_model)
        if self.model_name == 'resnet18':
            update_blocks = ['base', 'ds3', 'ds5', 'ds7', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']

        # updata all
        weights = copy.deepcopy(self.uploaded_samples)
        tot_samples = sum(weights)
        for i, w in enumerate(weights):
            weights[i] = w / tot_samples
        for block in update_blocks:
            glo = getattr(self.global_model, block)
            for param in glo.parameters():
                param.data.zero_()
            for w, client_model in zip(weights, self.uploaded_models):
                cli = getattr(client_model, block)  # client.model.base
                for server_param, client_param in zip(glo.parameters(), cli.parameters()):
                    server_param.data += client_param.data.clone() * w

        # optional updating
        column_sum = list(map(sum, zip(*self.freeze_infos)))
        # 对于每一个block
        for i in range(len(block_names)):
            # 有client更新
            if column_sum[i] != len(self.uploaded_models):
                block = getattr(self.global_model, block_names[i])   #self.global_model.base

                # 置为零
                for param in block.parameters():
                    param.data.zero_()

                uploaded_weights = []  #client的权重
                uploaded_blocks = []  #更新的模型
                total_num = 0
                # 对于每一个client
                for freeze_info, client_model, uploaded_samples in zip(self.freeze_infos, self.uploaded_models, self.uploaded_samples):
                    # 更新了当前的block
                    if freeze_info[i] == 0:
                        total_num += uploaded_samples
                        uploaded_weights.append(uploaded_samples)
                        client_block = getattr(client_model, block_names[i])  # client.model.base
                        uploaded_blocks.append(client_block)

                # 计算权重
                for j, m in enumerate(uploaded_weights):
                    uploaded_weights[j] = m / total_num

                # 更新全局模型
                for w, client_block in zip(uploaded_weights, uploaded_blocks):
                    for server_param, client_param in zip(block.parameters(), client_block.parameters()):
                        server_param.data += client_param.data.clone() * w


