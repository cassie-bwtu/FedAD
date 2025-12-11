import copy
import os
import torch
import numpy as np
import time
import random
import h5py
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data, read_test_data, get_info, get_info_test, get_flops, get_drop_info
import pandas as pd
import csv

class Server(object):
    def __init__(self, args, times):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.dataset_info = args.dataset_info
        self.model_name = args.model_name
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.clients_id = []
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_test_tg_acc = []
        self.rs_test_tg_auc = []
        self.rs_train_tg_loss = []
        self.rs_test_tgb_acc = []
        self.rs_test_tgb_auc = []
        self.rs_train_tgb_loss = []
        self.num_drop = args.num_drop
        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.batch_num_per_client = args.batch_num_per_client

        self.static = args.static
        self.ratio = args.ratio

        self.flops_rate = args.flops_rate
        self.flops_level = args.flops_level

        self.flops_level_list = [float(r) for r in self.flops_level.split(',')]

        self.client0_flops_level = args.client0_flops_level


    # clientObj: client类
    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients): #都是长度为#clients的列表
            train_data = read_client_data(self.dataset, self.dataset_info, i, is_train=True)
            test_data = read_client_data(self.dataset, self.dataset_info, i, is_train=False)
            flops_level = get_flops(i, self.flops_rate, self.flops_level)
            policy = get_drop_info(self.model_name, flops_level)

            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow,
                               policy=policy,
                               flops_level=flops_level)
            self.clients.append(client)


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        '''
        :param slow_rate:
        :return: []bool列表
        '''
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients


    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = \
            np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        self.clients_id.append([c.id for c in selected_clients])

        return selected_clients


    def load_test_data(self, batch_size=None):
        # ids = [i for i in range(self.num_clients)]
        ids = [i for i in range(self.num_clients)]
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_test_data(self.dataset, self.dataset_info, ids, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


    # send models to clients
    def send_models(self):
        assert (len(self.clients) > 0)

        # 发给每一个client
        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)   # *2


    # receive models from clients
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # selected_clients 当前轮参与的用户， len(selected_clients)=current_num_join_clients
        self.active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        # 成功参与训练的clients
        for client in self.active_clients:
            try:
                # client_time_cost平均时间开销
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        # 计算每个clients的样本权重
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # self.global_model = copy.deepcopy(self.global_model)
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        # 全局模型更新

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset, self.dataset_info)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + '_' + self.model_name + '_' + str(self.num_clients) + '_static' + str(self.static) + "_ratio" + str(self.ratio) + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset, self.dataset_info)
        model_path = os.path.join(model_path, self.algorithm + '_' + self.model_name + '_' + str(self.num_clients) + '_static' + str(self.static) + "_ratio" + str(self.ratio) + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset, self.dataset_info)
        model_path = os.path.join(model_path, self.algorithm + '_' + self.model_name + '_' + str(self.num_clients) + '_static' + str(self.static) + "_ratio" + str(self.ratio) + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.algorithm + "_" + self.model_name + '_' + str(self.num_clients) + '_static' + str(self.static) + "_ratio" + str(self.ratio)
        result_path = os.path.join("../results", self.dataset, self.dataset_info)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            path = algo + "_" + self.goal + "_" + str(self.times)
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
        algo = self.algorithm + "_" + self.model_name + '_' + str(self.num_clients) + '_static' + str(self.static) + "_ratio" + str(self.ratio)
        result_path = os.path.join("../results", self.dataset, self.dataset_info)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        # path = algo + "_selectedClients_" + self.goal + "_" + str(self.times)
        # file_path = result_path + "{}.csv".format(path)
        file_path = os.path.join(result_path, algo + "_selectedClients_" + self.goal + "_" + str(self.times) + ".csv")
        clients_id = []
        clients_traintime = []
        for client in self.active_clients:
            clients_id.append(client.id)
            clients_traintime.append(client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'])
        with open(file_path, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(clients_id)
            writer.writerow(clients_traintime)


    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            # cl=losses, ns=#samples
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses


    def test_metrics(self):
        testloaderfull = self.load_test_data()

        self.global_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
            #print(output)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc


    # evaluated all clients
    def evaluate(self, acc=None, auc=None, cacc=None, cauc=None, loss=None):
        acc_num, test_num, test_auc = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = acc_num * 1.0 / test_num  #平均accuracy（对于所有参与训练的样本）
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])  #平均损失
        # accs = [a / n for a, n in zip(stats[2], stats[1])]  #每一个clients的acc
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]  #每一个clients的auc

        '''
        client_accs = []
        client_aucs = []
        for i in range(len(ids)):
            client_accs.append(accs[i])
            client_aucs.append(aucs[i])
        '''

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

        # print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        # print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        # print("Test Accuracy for clients: {}".format(client_accs))
        # print("Test AUC for clients: {}".format(client_aucs))


    def evaluate_timegap(self):
        self.rs_test_tg_acc.append(self.rs_test_acc[-1])
        self.rs_test_tg_auc.append(self.rs_test_auc[-1])
        self.rs_train_tg_loss.append(self.rs_train_loss[-1])

        max_idx = np.argmax(np.array(self.rs_test_acc))
        self.rs_test_tgb_acc.append(self.rs_test_acc[max_idx])
        self.rs_test_tgb_auc.append(self.rs_test_auc[max_idx])
        self.rs_train_tgb_loss.append(self.rs_train_loss[max_idx])

        print("The current model")
        print("Averaged Train Loss: {:.4f}".format(self.rs_train_loss[-1]))
        print("Averaged Test Accurancy: {:.4f}".format(self.rs_test_acc[-1]))
        print("Averaged Test AUC: {:.4f}".format(self.rs_test_auc[-1]))

        print("Historical best model")
        print("Averaged Train Loss: {:.4f}".format(self.rs_train_loss[max_idx]))
        print("Averaged Test Accurancy: {:.4f}".format(self.rs_test_acc[max_idx]))
        print("Averaged Test AUC: {:.4f}".format(self.rs_test_auc[max_idx]))


    def save_loss(self):
        path = 'loss_test_res.txt'
        with open(path, "a+") as file:
            for item in self.rs_train_loss:
                file.write(item + " ")

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        '''
        :param acc_lss: test acc list (for each round)
        :param top_cnt: 超参数 100
        :param div_value:
        :return:
        '''
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            # 如果训练了top_cnt轮，都没超过最大值。就退出
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt  # maximum的索引
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


