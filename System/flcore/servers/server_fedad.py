import csv
import json
import os
import time

import h5py
import numpy as np
from flcore.clients.client_fedad import clientFedAD
from flcore.servers.serverbase import Server
from threading import Thread
import random
import copy
from line_profiler import LineProfiler
import psutil
import torch
from flcore.trainmodel.models import ReconstructRN18
from utils.data_utils import get_info_test


class FedAD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.global_policynet = copy.deepcopy(args.policynet)
        self.set_slow_clients()
        self.set_clients(clientFedAD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.model_name = args.model_name
        self.train_time_list = []

        self.model_flops = args.model_flops
        self.train_time_cons = self.model_flops / args.max_flops * args.avg_num_train
        self.test_time_cons = self.model_flops / args.max_flops * args.avg_num_test

        self.pn_percent = []

        self.add_cons = args.add_cons
        self.sigma = args.sigma
        self.time_gap = args.time_gap
        self.dynamic_change = args.dynamic
        self.change_gap = args.change_gap

        self.tau = args.tau

        self.records_dict = {}
        self.record = args.record


    def train(self):
        eval_times = 1
        for i in range(self.global_rounds + 1):

            if self.dynamic_change and i != 0 and i%self.change_gap == 0:
                for client in self.clients:
                    new_flops_level = np.random.choice(self.flops_level_list)
                    client.set_flops_level(new_flops_level)

            self.selected_clients = self.select_clients()

            self.send_models()  #client.model update

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train_batch(self.train_time_cons)
                torch.cuda.empty_cache()

            self.receive_models()
            self.aggregate_parameters()

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
        print(max(self.rs_test_acc))
        print("\nAverage time cost for training per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.record:
            self.save_records()


    def save_records(self):
        filename = f"../results/{self.dataset}_{self.model_name}_bs{self.batch_size}_PolicyRecords.h5"
        for client in self.clients:
            client_dict = {}
            for i, record in enumerate(client.client_records):
                client_dict[f"round_{i}"] = record
            self.records_dict[f"client_{client.id}"] = client_dict
        with h5py.File(filename, 'w') as hf:
            self._save_records_dict(hf, self.records_dict)

    def _save_records_dict(self, group, records):
        for client_id, rounds in records.items():
            client_group = group.create_group(client_id)
            for round_id, samples in rounds.items():
                round_group = client_group.create_group(round_id)
                json_data = json.dumps(samples)
                round_group.create_dataset("data", data=json_data)


    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.drop_infos = []
        self.uploaded_samples = []
        self.uploaded_models = []
        self.uploaded_policynets = []

        for client in active_clients:
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
                self.uploaded_policynets.append(client.policynet)


    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.global_model)
        self.global_policynet = copy.deepcopy(self.global_policynet)

        if self.model_name == 'resnet18':
            update_blocks = ['base', 'ds3', 'ds5', 'ds7', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']
        elif self.model_name == 'resnet34':
            update_blocks = ['base', 'ds4', 'ds8', 'ds14', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == 'resnet50':
            update_blocks = ['base', 'ds1', 'ds4', 'ds8', 'ds14', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == 'mobilenet_v2':
            update_blocks = ['base', 'block1', 'block2', 'block4', 'block7', 'block11', 'block14', 'block17', 'block18', 'head']
            block_names = ['block3', 'block5', 'block6', 'block8', 'block9', 'block10', 'block12', 'block13', 'block15', 'block16']
        elif self.model_name == 'regnet':
            update_blocks = ['base', 'ds1', 'ds2', 'ds3', 'ds7', 'head']
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13']
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
            if column_sum[i] < len(self.uploaded_models):
                server_block = getattr(self.global_model.blocks, block_names[i])  # self.global_model.block1

                # 置为零
                for name, param in server_block.state_dict().items():
                    param.data.zero_()

                uploaded_weights = []  # client的权重
                uploaded_blocks = []  # 更新的模型
                # 对于每一个client
                for drop_info, client_model, uploaded_samples in zip(self.drop_infos, self.uploaded_models, self.uploaded_samples):
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


        for name, param in self.global_policynet.state_dict().items():
            param.data.zero_()
        for w, client_policynet in zip(weights, self.uploaded_policynets):
            for (server_name, server_param), (client_name, client_param) in zip(self.global_policynet.state_dict().items(),
                                                                                client_policynet.state_dict().items()):
                if 'num_batches_tracked' in server_name:
                    server_param.data += int(client_param.data.clone() * w)
                else:
                    server_param.data += client_param.data.clone() * w


    def send_models(self):
        assert (len(self.clients) > 0)

        # 发给每一个client
        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model, self.global_policynet)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)   # *2


    def local_evaluate(self, acc=None, auc=None, cacc=None, cauc=None, loss=None):
        stats = self.local_test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])  #平均损失
        accs = [a / n for a, n in zip(stats[2], stats[1])]  #每一个clients的acc
        aucs = [a / n for a, n in zip(stats[3], stats[1])]  #每一个clients的auc

        if acc == None:
            # 第一轮
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if auc == None:
            self.rs_test_auc.append(test_auc)
        else:
            auc.append(test_auc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))


    def local_test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics(self.test_time_cons)
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc


    def save_global_model(self):
        model_path = os.path.join("models", self.dataset, self.dataset_info)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + '_' + self.model_name + '_sigma' + str(self.sigma) + '_tau' + str(self.tau) + '_ac' + str(self.add_cons) + "_server" + ".pt")
        torch.save(self.global_model, model_path)


    def save_results(self):
        algo = self.algorithm + "_" + self.model_name + '_sigma' + str(self.sigma) + '_tau' + str(self.tau) + '_ac' + str(self.add_cons) + "_" + self.goal + "_" + str(self.times)
        result_path = os.path.join("../results", self.dataset, self.dataset_info)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            path = algo + "_" + self.goal + "_" + str(self.times) + '_changegap' + str(self.change_gap)
            file_path = os.path.join(result_path, path + ".h5")
            # file_path = result_path + "{}.h5".format(path)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_time_cost', data=self.Budget)

                hf.create_dataset('rs_test_tg_acc', data=self.rs_test_tg_acc)
                hf.create_dataset('rs_test_tg_auc', data=self.rs_test_tg_auc)
                hf.create_dataset('rs_train_tg_loss', data=self.rs_train_tg_loss)

                hf.create_dataset('rs_test_tgb_acc', data=self.rs_test_tgb_acc)
                hf.create_dataset('rs_test_tgb_auc', data=self.rs_test_tgb_auc)
                hf.create_dataset('rs_train_tgb_loss', data=self.rs_train_tgb_loss)


    def save_cost(self):
        # 每个用户每轮的训练时间
        algo = self.algorithm + "_" + self.model_name + '_sigma' + str(self.sigma) + '_tau' + str(self.tau) + '_ac' + str(self.add_cons)
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

