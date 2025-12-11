import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client

class clientFreeze(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.model_name = args.model_name
        self.learning_rate = args.local_learning_rate
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.freeze_info = kwargs['info']

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        if self.model_name == 'resnet18':
            block_names = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']
        for i, block_name in enumerate(block_names):
            component_model = getattr(self.model, block_name)
            for param in component_model.parameters():
                param.requires_grad = not self.freeze_info[i]

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.learning_rate_decay_gamma
        )

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1  #num_rounds 轮数
        self.train_time_cost['total_cost'] += time.time() - start_time  #训练总时间

