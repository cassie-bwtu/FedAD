import copy
import os

import torch
import numpy as np
import time
import torch.nn as nn
from flcore.clients.clientbase import Client
from flcore.trainmodel.models import BlockSplitRN18, ReconstructRN18, ReconstructRN34, ReconstructRN50, NewPolicyRN18, NewPolicyRN34, NewPolicyRN50, ReconstructRN6, NewPolicyRN6, ReconstructRN10, NewPolicyRN10, \
    ReconstructRegNet, ReconstructMN2, NewPolicyRegNet, NewPolicyMN2, ReconstructViT, NewPolicyViT, BlockSplitDistilBERT, NewPolicyDistilBERT, ReconstructDistilBERT
from utils.data_utils import get_info, get_info_test, get_drop_info
from memory_profiler import profile
import psutil
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class clientDrop(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.model_name = args.model_name
        # self.id = id
        # self.num_drop = args.num_drop
        self.learning_rate = args.local_learning_rate
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.policy = kwargs['policy']

        # self.drop_info = get_info_test(args.model_name, args.num)  #加的

        self.ratio = args.ratio

        # self.num = args.num  #加的

        self.static = args.static
        self.roundchange = args.roundchange
        self.model = copy.deepcopy(args.model)
        self.loss = nn.CrossEntropyLoss()

        self.flops_level = kwargs['flops_level']

        # droppable
        if self.model_name == 'resnet18':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']
        elif self.model_name == 'resnet34':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == 'resnet50':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16']
        elif self.model_name == 'resnet6':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']
        elif self.model_name == 'resnet10':
            self.block_names = ['block1', 'block2', 'block3', 'block4']
        elif self.model_name == 'regnet':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9',
                           'block10', 'block11', 'block12', 'block13']
        elif self.model_name == 'mobilenet_v2':
            self.block_names = ['block3', 'block5', 'block6', 'block8', 'block9', 'block10', 'block12', 'block13', 'block15',
                           'block16']
        elif self.model_name == 'vit':
            self.block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
        elif self.model_name == 'bert':
            self.block_names = ['block1', 'block2', 'block3', 'block4']

        # self.sub_model = copy.deepcopy(ReconstructRN18(self.model, infos=self.drop_info))
        # self.optimizer = torch.optim.SGD(self.sub_model.parameters(), lr=self.learning_rate)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma
        # )

    # @profile(precision=4, stream=open("profile/memory_profiler.log", "a+"))
    def train(self):
        start_time = time.time()

        trainloader = self.load_train_data()

        if self.static == True and self.roundchange == True:
            self.drop_info = self.policy[:, 1].tolist()
            model = copy.deepcopy(self.model)
            if self.model_name == 'resnet18':
                self.sub_model = copy.deepcopy(ReconstructRN18(model, infos=self.drop_info))
            elif self.model_name == 'resnet34':
                self.sub_model = copy.deepcopy(ReconstructRN34(model, infos=self.drop_info))
            elif self.model_name == 'resnet50':
                self.sub_model = copy.deepcopy(ReconstructRN50(model, infos=self.drop_info))
            elif self.model_name == 'resnet6':
                self.sub_model = copy.deepcopy(ReconstructRN6(model, infos=self.drop_info))
            elif self.model_name == 'resnet10':
                self.sub_model = copy.deepcopy(ReconstructRN10(model, infos=self.drop_info))
            elif self.model_name == 'regnet':
                self.sub_model = copy.deepcopy(ReconstructRegNet(model, infos=self.drop_info))
            elif self.model_name == 'mobilenet_v2':
                self.sub_model = copy.deepcopy(ReconstructMN2(model, infos=self.drop_info))
            elif self.model_name == 'vit':
                self.sub_model = copy.deepcopy(ReconstructViT(model, infos=self.drop_info))
            elif self.model_name == 'bert':
                self.sub_model = copy.deepcopy(ReconstructDistilBERT(model, infos=self.drop_info))


            self.optimizer = torch.optim.SGD(self.sub_model.parameters(), lr=self.learning_rate)
            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=self.learning_rate_decay_gamma
            )

            self.sub_model.train()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for epoch in range(max_local_epochs):
                losses = 0
                num = 0
                for i, (x, y) in enumerate(trainloader):
                    # st = time.time()
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.sub_model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    losses += loss.item() * y.shape[0]
                    num += y.shape[0]
                    # et = time.time() - st
                    # print("training time for one batch: {}".format(et))
                # print(losses/num)

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1  # num_rounds 轮数
            self.train_time_cost['total_cost'] += time.time() - start_time  # 训练总时间


        if self.static == False and self.roundchange == True:
            policy = get_drop_info(self.model_name, self.flops_level).to(self.device)
            self.drop_info = policy[:, 1].tolist()
            model = copy.deepcopy(self.model)
            if self.model_name == 'resnet18':
                self.sub_model = copy.deepcopy(ReconstructRN18(model, infos=self.drop_info))
            elif self.model_name == 'resnet34':
                self.sub_model = copy.deepcopy(ReconstructRN34(model, infos=self.drop_info))
            elif self.model_name == 'resnet50':
                self.sub_model = copy.deepcopy(ReconstructRN50(model, infos=self.drop_info))
            elif self.model_name == 'resnet6':
                self.sub_model = copy.deepcopy(ReconstructRN6(model, infos=self.drop_info))
            elif self.model_name == 'resnet10':
                self.sub_model = copy.deepcopy(ReconstructRN10(model, infos=self.drop_info))
            elif self.model_name == 'regnet':
                self.sub_model = copy.deepcopy(ReconstructRegNet(model, infos=self.drop_info))
            elif self.model_name == 'mobilenet_v2':
                self.sub_model = copy.deepcopy(ReconstructMN2(model, infos=self.drop_info))
            elif self.model_name == 'vit':
                self.sub_model = copy.deepcopy(ReconstructViT(model, infos=self.drop_info))
            elif self.model_name == 'bert':
                self.sub_model = copy.deepcopy(ReconstructDistilBERT(model, infos=self.drop_info))


            self.optimizer = torch.optim.SGD(self.sub_model.parameters(), lr=self.learning_rate)
            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=self.learning_rate_decay_gamma
            )

            self.sub_model.train()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for epoch in range(max_local_epochs):
                losses = 0
                num = 0
                for i, (x, y) in enumerate(trainloader):
                    # st = time.time()
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.sub_model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    losses += loss.item() * y.shape[0]
                    num += y.shape[0]
                    # et = time.time() - st
                    # print("training time for one batch: {}".format(et))
                # print(losses/num)

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1  # num_rounds 轮数
            self.train_time_cost['total_cost'] += time.time() - start_time  # 训练总时间


        elif self.static == False and self.roundchange == False:
            if self.model_name == 'resnet18':
                self.batch_model = NewPolicyRN18(copy.deepcopy(self.model))
            elif self.model_name == 'resnet34':
                self.batch_model = NewPolicyRN34(copy.deepcopy(self.model))
            elif self.model_name == 'resnet50':
                self.batch_model = NewPolicyRN50(copy.deepcopy(self.model))
            elif self.model_name == 'resnet6':
                self.batch_model = NewPolicyRN6(copy.deepcopy(self.model))
            elif self.model_name == 'resnet10':
                self.batch_model = NewPolicyRN10(copy.deepcopy(self.model))
            elif self.model_name == 'regnet':
                self.batch_model = NewPolicyRegNet(copy.deepcopy(self.model))
            elif self.model_name == 'mobilenet_v2':
                self.batch_model = NewPolicyMN2(copy.deepcopy(self.model))
            elif self.model_name == 'vit':
                self.batch_model = NewPolicyViT(copy.deepcopy(self.model))
            elif self.model_name == 'bert':
                self.batch_model = NewPolicyDistilBERT(copy.deepcopy(self.model))

            self.batch_model.train()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            self.update_time = [0 for m in range(len(self.block_names))]

            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    policy = get_drop_info(self.model_name, self.flops_level).to(self.device)
                    self.update_time = [a + b for a, b in zip(self.update_time, policy[:, 0].tolist())]

                    # time_cost = (self.basic_flops + torch.sum(policy[:,0] * self.block_flops.to(self.device))) * self.train_samples * 2 / self.flops
                    output = self.batch_model(x, policy)

                    for param in self.batch_model.parameters():
                        param.requires_grad = True
                    for i in range(policy.shape[0]):
                        if policy[i, 1] == 1:
                            drop = getattr(self.batch_model.blocks, f'block{i + 1}')
                            for param in drop.parameters():
                                param.requires_grad = False

                    parameters_to_optimize = filter(lambda p: p.requires_grad, self.batch_model.parameters())


                    self.optimizer = torch.optim.SGD(parameters_to_optimize, lr=self.learning_rate)
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=self.optimizer,
                        gamma=self.learning_rate_decay_gamma
                    )

                    # together
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1  # num_rounds 轮数
            self.train_time_cost['total_cost'] += time.time() - start_time  # 训练总时间

            self.drop_info = [1 if n == 0 else 0 for n in self.update_time]
            if self.model_name == 'resnet18':
                self.sub_model = copy.deepcopy(ReconstructRN18(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'resnet34':
                self.sub_model = copy.deepcopy(ReconstructRN34(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'resnet50':
                self.sub_model = copy.deepcopy(ReconstructRN50(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'resnet6':
                self.sub_model = copy.deepcopy(ReconstructRN6(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'resnet10':
                self.sub_model = copy.deepcopy(ReconstructRN10(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'regnet':
                self.sub_model = copy.deepcopy(ReconstructRegNet(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'mobilenet_v2':
                self.sub_model = copy.deepcopy(ReconstructMN2(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'vit':
                self.sub_model = copy.deepcopy(ReconstructViT(copy.deepcopy(self.batch_model), infos=self.drop_info))
            elif self.model_name == 'bert':
                self.sub_model = copy.deepcopy(ReconstructDistilBERT(copy.deepcopy(self.batch_model), infos=self.drop_info))

        torch.cuda.empty_cache()


        # self.train_time_cost['num_rounds'] += 1  #num_rounds 轮数
        # self.train_time_cost['total_cost'] += time.time() - start_time  #训练总时间
        # print("training time for one epoch: {}".format(time.time() - start_time))


    def set_parameters(self, model):
        '''
        :param model: global model
        :return:
        '''
        # for (server_name, new_param), (client_name, old_param) in zip(model.state_dict().items(), self.model.state_dict().items()):
        #     old_param.data = new_param.data.clone()
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):   #self.model: BlockSplitRN18()
        #     old_param.data = new_param.data.clone()
        self.model.load_state_dict(model.state_dict())

