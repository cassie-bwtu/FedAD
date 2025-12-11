import models
import os
from ptflops import get_model_complexity_info
import torch.nn as nn
import torch
from utils.gumbel_softmax import get_policy
from torch.nn.functional import softmax, softplus
from flcore.trainmodel.policy_network import *
from regnet import *

import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, device, eps=1e-20):
    # U = torch.cuda.FloatTensor(shape).uniform_()   #cuda
    U = torch.FloatTensor(shape).uniform_().to(device)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, device, tau=1):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / tau, dim=-1)

def gumbel_softmax(logits, device, tau=1, hard=True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, device, tau)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard, y


# class BlockSplitRegNet(nn.Module):
#     def __init__(self, RegNet):
#         super(BlockSplitRegNet, self).__init__()
#
#         num_layers = 13
#         self.blocks = nn.ModuleDict()
#         self.blocks['base'] = RegNet.stem
#         for i in range(num_layers):
#             if i == 0:
#                 self.blocks[f'block{(i + 1)}'] = RegNet.s1.b1
#             elif i == 1:
#                 self.blocks[f'block{(i + 1)}'] = RegNet.s2.b1
#             elif i > 1 and i <= 5:
#                 self.blocks[f'block{(i + 1)}'] = getattr(RegNet.s3, f'b{i-1}')
#             else:
#                 self.blocks[f'block{(i + 1)}'] = getattr(RegNet.s4, f'b{i-5}')
#         self.blocks['head'] = RegNet.head
#
#     def forward(self, x):
#         out = x
#         for block_name, block in self.blocks.items():
#             out = block(out)
#
#         return out


# class ReconstructRegNet(nn.Module):
#     def __init__(self, RegNet, infos=None):
#         super(ReconstructRegNet, self).__init__()
#         if infos is None:
#             infos = [0 for i in range(9)]
#         num_blocks = 13
#         self.keep_blocks = ['block1', 'block2', 'block3', 'block7']
#         self.drop_blocks = ['block4', 'block5', 'block6', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13']
#
#         self.blocks = nn.ModuleDict()
#         self.blocks['base'] = RegNet.blocks.base
#
#         for i in range(num_blocks):
#             if f'block{i + 1}' in self.keep_blocks:
#                 self.blocks[f'block{i + 1}'] = getattr(RegNet.blocks, f'block{i + 1}')
#             elif f'block{i + 1}' in self.drop_blocks and infos[self.drop_blocks.index(f'block{i + 1}')] == 0:
#                 self.blocks[f'block{i + 1}'] = getattr(RegNet.blocks, f'block{i + 1}')
#
#         self.blocks['head'] = RegNet.blocks.head
#
#     def forward(self, x):
#         out = x
#         for block_name, block in self.blocks.items():
#             out = block(out)
#
#         return out


# class NewPolicyRegNet(nn.Module):
#     def __init__(self, RegNet):
#         super(NewPolicyRegNet, self).__init__()
#
#         self.blocks = RegNet.blocks
#         self.keep_blocks = ['block1', 'block2', 'block3', 'block7']
#         self.drop_blocks = ['block4', 'block5', 'block6', 'block8', 'block9', 'block10', 'block11', 'block12',
#                             'block13']
#         self.num_blocks = 13
#
#     def forward(self, x, policy):
#         # (9,2) -- > (9,)
#         policy = policy[:, 0]  # reserve
#         out = x
#         out = self.blocks['base'](out)
#
#         for i in range(self.num_blocks):
#             if f'block{i + 1}' in self.keep_blocks:
#                 out = self.blocks[f'block{i + 1}'](out)
#             elif f'block{i + 1}' in self.drop_blocks:
#                 idx = self.drop_blocks.index(f'block{i + 1}')
#                 action = policy[idx].contiguous()
#                 action_mask = action.float().view(-1, 1, 1, 1)
#                 # version 1
#                 if action.item() == 1:   #reserve
#                     out = self.blocks[f'block{i + 1}'](out) * action_mask
#                 # version 2
#                 # out_block = self.blocks[f'block{i + 1}'](out)
#                 # out = out_block * action_mask + out * (1 - action_mask)
#
#         out = self.blocks['head'](out)
#
#         return out


class BlockSplitRegNet(nn.Module):
    def __init__(self, RegNet):
        super(BlockSplitRegNet, self).__init__()

        num_layers = 13
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = RegNet.stem
        for i in range(num_layers):
            if i == 0:
                self.blocks[f'ds{(i + 1)}'] = nn.Sequential(RegNet.s1.b1.proj, RegNet.s1.b1.bn)
                self.blocks[f'block{(i + 1)}'] = RegNet.s1.b1.f
            elif i == 1:
                self.blocks[f'ds{(i + 1)}'] = nn.Sequential(RegNet.s2.b1.proj, RegNet.s2.b1.bn)
                self.blocks[f'block{(i + 1)}'] = RegNet.s2.b1.f
            elif 1 < i <= 5:
                if i == 2:
                    self.blocks[f'ds{(i + 1)}'] = nn.Sequential(RegNet.s3.b1.proj, RegNet.s3.b1.bn)
                    self.blocks[f'block{(i + 1)}'] = RegNet.s3.b1.f
                else:
                    self.blocks[f'block{(i + 1)}'] = getattr(RegNet.s3, f'b{i-1}')
            else:
                if i == 6:
                    self.blocks[f'ds{(i + 1)}'] = nn.Sequential(RegNet.s4.b1.proj, RegNet.s4.b1.bn)
                    self.blocks[f'block{(i + 1)}'] = RegNet.s4.b1.f
                else:
                    self.blocks[f'block{(i + 1)}'] = getattr(RegNet.s4, f'b{i-5}')
        self.blocks['head'] = RegNet.head
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        flag = 0
        for block_name, block in self.blocks.items():
            if flag == 1:
                flag = 0
                continue
            if 'base' in block_name or 'head' in block_name:
                out = block(out)
            elif 'ds' in block_name and block_name.replace('ds', 'block') in self.blocks:
                identify = block(out)
                out = self.relu(self.blocks[block_name.replace('ds', 'block')](out) + identify)
                flag = 1
            elif 'ds' in block_name and block_name.replace('ds', 'block') not in self.blocks:
                out = self.relu(block(out))
            else:
                identify = out
                out = self.relu(block(out) + identify)

        return out


class ReconstructRegNet(nn.Module):
    def __init__(self, regnet, infos=None):
        super(ReconstructRegNet, self).__init__()
        if infos is None:
            infos = [0 for i in range(13)]
        ds_idx = [1, 2, 3, 7]

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = regnet.blocks.base

        for i in range(len(infos)):
            if infos[i] == 0:
                if (i+1) not in ds_idx:
                    self.blocks[f'block{i + 1}'] = getattr(regnet.blocks, f'block{i + 1}')
                else:
                    self.blocks[f'ds{i + 1}'] = getattr(regnet.blocks, f'ds{i + 1}')
                    self.blocks[f'block{i + 1}'] = getattr(regnet.blocks, f'block{i + 1}')
            elif infos[i] == 1:
                if (i+1) in ds_idx:
                    self.blocks[f'ds{i + 1}'] = getattr(regnet.blocks, f'ds{i + 1}')

        self.blocks['head'] = regnet.blocks.head

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        flag = 0
        for block_name, block in self.blocks.items():
            if flag == 1:
                flag = 0
                continue
            if 'base' in block_name or 'head' in block_name:
                out = block(out)
            elif 'ds' in block_name and block_name.replace('ds', 'block') in self.blocks:
                identify = block(out)
                out = self.relu(self.blocks[block_name.replace('ds', 'block')](out) + identify)
                flag = 1
            elif 'ds' in block_name and block_name.replace('ds', 'block') not in self.blocks:
                out = self.relu(block(out))
            else:
                identify = out
                out = self.relu(block(out) + identify)

        return out


class NewPolicyRegNet(nn.Module):
    def __init__(self, regnet):
        super(NewPolicyRegNet, self).__init__()

        self.blocks = regnet.blocks
        self.ds_idx = [1, 2, 3, 7]

    def forward(self, x, policy):
        # (16,2) -- > (16,)
        policy = policy[:, 0]  #reserve
        out = x
        out = self.blocks['base'](out)

        for i in range(policy.shape[0]):
            action = policy[i].contiguous()
            action_mask = action.float().view(-1, 1, 1, 1)
            if (i + 1) in self.ds_idx:
                residual = self.blocks[f'ds{i + 1}'](out)
            else:
                residual = out

            if action.item() == 1:
                out = F.relu(residual + self.blocks[f'block{i + 1}'](out)) * action_mask
            else:
                out = residual * (1 - action_mask)

        out = self.blocks['head'](out)

        return out




# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
# model = models.__dict__['regnet_200m']()
# model = model.cuda()
#
# print(model)

# flops, params = get_model_complexity_info(model, (3,32,32), as_strings=True, print_per_layer_stat=True)
# print("%s |%s" % (flops, params))


if __name__ == '__main__':
    model = regnet_200m()
    # print(model)
    splitRN = BlockSplitRegNet(model)
    # print(splitRN)
    # x = torch.ones(10, 3, 32, 32)
    # y = splitRN(x)
    # print(y.shape)

    #
    flops, params = get_model_complexity_info(splitRN, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
    print("%s |%s" % (flops, params))

    # x = torch.ones(10, 3, 32, 32)
    # y = splitRN(x)
    # print(y.shape)
    #
    # newRN = ReconstructRegNet(splitRN, infos=[1,0,0,1,0,0,0,0,0,0,1,1,1])
    # x = torch.ones(10, 3, 32, 32)
    # y = newRN(x)
    # print(newRN)
    # print(y.shape)

    # model = regnet_200m()
    #
    # drop_blocks = ['block4', 'block5', 'block6', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13']
    # splitRegNet = BlockSplitRegNet(model)
    # batch_model = NewPolicyRegNet(splitRegNet)
    # policy_network = policynet(num_classes=9 * 2)
    #
    # optimizer_pn = torch.optim.SGD(policy_network.parameters(), lr=5)
    # learning_rate_scheduler_pn = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer=optimizer_pn,
    #     gamma=0.1
    # )
    # loss_acc = nn.CrossEntropyLoss()
    # device = "cpu"
    #
    # for epoch in range(10):
    #     x = torch.rand(10, 3, 32, 32)
    #     y = torch.randint(0, 100, (10,))
    #     probs = policy_network(x)
    #     poli = softmax(torch.mean(probs, dim=0).view(-1, 2), dim=1)
    #     poli, probs = gumbel_softmax(poli, device, tau=0.6, hard=True)
    #     block_flops = torch.tensor([2, 3, 1, 4, 5, 2, 3, 5, 4])
    #     mask = get_policy(probs, block_flops, 20)
    #     policy = (mask - poli).detach() + poli
    #
    #     output = batch_model(x, policy)
    #
    #     for param in batch_model.parameters():
    #         param.requires_grad = True
    #     for i in range(policy.shape[0]):
    #         if policy[i, 1] == 1:
    #             drop = getattr(batch_model.blocks, drop_blocks[i])
    #             for param in drop.parameters():
    #                 param.requires_grad = False
    #
    #     parameters_to_optimize = filter(lambda p: p.requires_grad, batch_model.parameters())
    #     optimizer_n = torch.optim.SGD(parameters_to_optimize, lr=10)
    #     learning_rate_scheduler_n = torch.optim.lr_scheduler.ExponentialLR(
    #         optimizer=optimizer_n,
    #         gamma=0.1
    #     )
    #
    #     # together
    #     loss = loss_acc(output, y)
    #     # loss_t = self.loss_t(time_cost, time_cons)
    #     # loss = loss_acc + self.alpha * loss_t
    #     optimizer_pn.zero_grad()
    #     optimizer_n.zero_grad()
    #     loss.backward()
    #     optimizer_n.step()
    #     optimizer_pn.step()

    # model = regnet_200m()
    #
    # splitRegNet = BlockSplitRegNet(model)
    #
    # loss_acc = nn.CrossEntropyLoss()
    # optimizer_n = torch.optim.SGD(model.parameters(), lr=0.1)
    # learning_rate_scheduler_n = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer=optimizer_n,
    #     gamma=0.1
    # )
    #
    # for epoch in range(10):
    #     x = torch.rand(10, 3, 32, 32)
    #     y = torch.randint(0, 100, (10,))
    #
    #     output = splitRegNet(x)
    #
    #     # together
    #     loss = loss_acc(output, y)
    #     # loss_t = self.loss_t(time_cost, time_cons)
    #     # loss = loss_acc + self.alpha * loss_t
    #     optimizer_n.zero_grad()
    #     loss.backward()
    #     optimizer_n.step()
