import copy
import gc
import os
import random

import torch
import numpy as np
import time
import torch.nn as nn
from sklearn.preprocessing import label_binarize
# from torch.nn.functional import gumbel_softmax, softmax
from torch.nn.functional import softmax, softplus
from flcore.clients.clientbase import Client
from flcore.trainmodel.models import BlockSplitRN18, ReconstructRN18, BatchRN18, PolicyRN18, SubRN18, NewPolicyRN18, BlockSplitRN34, ReconstructRN34, PolicyRN34, NewPolicyRN34, NewPolicyRN50, ReconstructRN50, NewPolicyMN2, ReconstructMN2, \
    NewPolicyRegNet, ReconstructRegNet, NewPolicyViT, ReconstructViT, BlockSplitDistilBERT, NewPolicyDistilBERT, ReconstructDistilBERT
from utils.gumbel_softmax import gumbel_softmax, get_mask, get_mask_new, get_policy
from sklearn import metrics
from torch.utils.data import DataLoader, ConcatDataset
from utils.data_utils import get_info, get_info_test, get_flops
from memory_profiler import profile
import psutil
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class clientFedAD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.model_name = args.model_name
        self.id = id
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.policy_network_learning_rate = args.policy_network_learning_rate
        self.flops_level = kwargs['flops_level']
        self.max_flops = args.max_flops
        self.flops = self.flops_level * self.max_flops
        self.model = copy.deepcopy(args.model)
        self.policynet = copy.deepcopy(args.policynet)
        self.loss_acc = nn.CrossEntropyLoss()
        self.loss_t = nn.MSELoss()

        self.optimizer_pn = torch.optim.SGD(self.policynet.parameters(), lr=self.policy_network_learning_rate)
        self.learning_rate_scheduler_pn = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_pn,
            gamma=self.learning_rate_decay_gamma
        )

        self.block_flops = args.block_flops
        self.basic_flops = args.basic_flops
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.sigma = args.sigma
        self.tau = args.tau
        self.add_cons = args.add_cons
        self.add_cons_test = args.add_cons_test
        self.mem_usage = []
        self.record = args.record
        self.client_records = []

        if self.model_name == 'resnet18':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']
        elif self.model_name == 'resnet34':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == 'resnet50':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == "mobilenet_v2":
            self.block_names = ['block3', 'block5', 'block6', 'block8', 'block9', 'block10', 'block12', 'block13', 'block15', 'block16']
        elif self.model_name == "regnet":
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13']
        elif self.model_name == 'vit':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
        elif self.model_name == 'bert':
            self.block_names = ['block1', 'block2', 'block3', 'block4']


    def train_batch(self, time_cons):
        trainloader = self.load_train_data()
        if self.model_name == 'resnet18':
            self.batch_model = NewPolicyRN18(copy.deepcopy(self.model))
        elif self.model_name == 'resnet34':
            self.batch_model = NewPolicyRN34(copy.deepcopy(self.model))
        elif self.model_name == 'resnet50':
            self.batch_model = NewPolicyRN50(copy.deepcopy(self.model))
        elif self.model_name == 'mobilenet_v2':
            self.batch_model = NewPolicyMN2(copy.deepcopy(self.model))
        elif self.model_name == 'regnet':
            self.batch_model = NewPolicyRegNet(copy.deepcopy(self.model))
        elif self.model_name == 'vit':
            self.batch_model = NewPolicyViT(copy.deepcopy(self.model))
        elif self.model_name == 'bert':
            self.batch_model = NewPolicyDistilBERT(copy.deepcopy(self.model))

        time_cons = torch.tensor(time_cons).to(self.device)
        self.ctbt = self.flops / self.train_samples
        flops_cons = time_cons * self.ctbt - self.basic_flops
        v = torch.tensor([[self.sigma, -1 * self.sigma] for i in range(len(self.block_names))]).to(self.device)

        self.batch_model.train()
        self.policynet.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        self.update_time = [0 for m in range(len(self.block_names))]
        reserve_list = []

        start_time = time.time()

        if self.record:
            round_records = []

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                probs = self.policynet(x)  # (batch,16)
                poli = torch.add(softmax(torch.mean(probs, dim=0).view(-1, 2), dim=1), v)
                probs = softmax(poli, dim=1)   #softmax

                if self.add_cons:
                    mask = get_policy(probs, self.block_flops, flops_cons).to(self.device)
                    policy = (mask - probs).detach() + probs

                else:
                    policy = poli  #gumbel

                if self.record:
                    round_records.append({
                        "batch_id": i,
                        # "label": y[0].item(),
                        "label": y.tolist(),
                        "drop_policy_str": ''.join(map(str, policy[:, 1].to(torch.long).tolist()))  # 保存成 "011" 形式
                    })

                self.update_time = [a + b for a, b in zip(self.update_time, policy[:, 0].tolist())]
                reserve_list.append(np.sum(policy[:, 0].tolist()))

                output = self.batch_model(x, policy)

                for param in self.batch_model.parameters():
                    param.requires_grad = True
                for i in range(policy.shape[0]):
                    if policy[i,1] == 1:
                        drop = getattr(self.batch_model.blocks, f'block{i + 1}')
                        for param in drop.parameters():
                            param.requires_grad = False

                parameters_to_optimize = filter(lambda p: p.requires_grad, self.batch_model.parameters())
                self.optimizer_n = torch.optim.SGD(parameters_to_optimize, lr=self.learning_rate)
                self.learning_rate_scheduler_n = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer_n,
                    gamma=self.learning_rate_decay_gamma
                )

                # together
                loss = self.loss_acc(output, y)
                self.optimizer_pn.zero_grad()
                self.optimizer_n.zero_grad()
                loss.backward()
                self.optimizer_n.step()
                self.optimizer_pn.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler_n.step()

        self.train_time_cost['num_rounds'] += 1  # num_rounds 轮数
        self.train_time_cost['total_cost'] += time.time() - start_time - 0.25  # 训练总时间

        if self.record:
            self.client_records.append(round_records)

        self.drop_info = [1 if n == 0 else 0 for n in self.update_time]
        self.num_reserve = np.mean(reserve_list)
        if self.model_name == 'resnet18':
            self.sub_model = copy.deepcopy(ReconstructRN18(copy.deepcopy(self.batch_model), infos=self.drop_info))
        elif self.model_name == 'resnet34':
            self.sub_model = copy.deepcopy(ReconstructRN34(copy.deepcopy(self.batch_model), infos=self.drop_info))
        elif self.model_name == 'resnet50':
            self.sub_model = copy.deepcopy(ReconstructRN50(copy.deepcopy(self.batch_model), infos=self.drop_info))
        elif self.model_name == 'mobilenet_v2':
            self.sub_model = copy.deepcopy(ReconstructMN2(copy.deepcopy(self.batch_model), infos=self.drop_info))
        elif self.model_name == 'regnet':
            self.sub_model = copy.deepcopy(ReconstructRegNet(copy.deepcopy(self.batch_model), infos=self.drop_info))
        elif self.model_name == 'vit':
            self.sub_model = copy.deepcopy(ReconstructViT(copy.deepcopy(self.batch_model), infos=self.drop_info))
        elif self.model_name == 'bert':
            self.sub_model = copy.deepcopy(ReconstructDistilBERT(copy.deepcopy(self.batch_model), infos=self.drop_info))

        torch.cuda.empty_cache()


    def set_parameters(self, model, policynet):
        '''
        :param model: global model
        :return:
        '''
        self.model.load_state_dict(model.state_dict())
        self.policynet.load_state_dict(policynet.state_dict())


    def test_metrics(self, time_cons):
        testloaderfull = self.load_test_data()
        self.batch_model = PolicyRN18(copy.deepcopy(self.model))
        time_cons = torch.tensor(time_cons).to(self.device)
        flops_cons = time_cons * self.flops / self.test_samples - self.basic_flops
        v = torch.tensor([[self.sigma, -1 * self.sigma] for i in range(len(self.block_names))]).to(self.device)

        self.policynet.eval()
        self.batch_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x,y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                probs = self.policynet(x)  # (batch,16)
                poli = softmax(torch.mean(probs, dim=0).view(-1, 2), dim=1) + v
                poli, probs = gumbel_softmax(poli, tau=0.6, hard=True)

                if self.add_cons_test:
                    mask = get_mask(poli, probs, self.block_flops, flops_cons, self.max_num_blocks)
                    policy = mask
                else:
                    policy = poli

                output = self.batch_model(x, policy)

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

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc


    def set_flops_level(self, flops_level):
        self.flops_level = flops_level
        self.flops = self.flops_level * self.max_flops
