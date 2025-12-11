from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from utils.gumbel_softmax import get_policy
from torch.nn.functional import softmax, softplus
from flcore.trainmodel.policy_network import *

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


# model = EfficientNet.from_name('efficientnet-b0')

# tiny-imagenet
# num_classes = 200  # Example: Set to 10 for 10 output classes
# model._fc = torch.nn.Linear(in_features=model._fc.in_features, out_features=num_classes)
# x = torch.ones(10,3,64,64)
# y = model(x)

# cifar100
# model._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=False)
# num_classes = 100  # Example: Set to 10 for 10 output classes
# model._fc = nn.Linear(in_features=model._fc.in_features, out_features=num_classes)
# x = torch.ones(10,3,32,32)
# y = model(x)

# emnist
# model._conv_stem = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# num_classes = 62  # Example: Set to 10 for 10 output classes
# model._fc = torch.nn.Linear(in_features=model._fc.in_features, out_features=num_classes)
# x = torch.ones(10,1,28,28)
# y = model(x)

# print(model)


# https://github.com/lukemelas/EfficientNet-PyTorch


class BlockSplitEfficientNet(nn.Module):
    def __init__(self, EfficientNet):
        super(BlockSplitEfficientNet, self).__init__()

        num_layers = 16
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = nn.Sequential(EfficientNet._conv_stem, EfficientNet._bn0)
        for i in range(num_layers):
            # module = getattr(MN2.features, f'{(i+1)}')
            self.blocks[f'block{(i+1)}'] = EfficientNet._blocks[i]
        # module = getattr(MN2.features, '18')
        # self.blocks['head'] = nn.Sequential(module, MN2.dropout, nn.Flatten(), MN2.fc)
        # avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.blocks['head'] = nn.Sequential(EfficientNet._conv_head, EfficientNet._bn1, EfficientNet._avg_pooling,
                                            EfficientNet._dropout, nn.Flatten(), EfficientNet._fc, EfficientNet._swish)
        # self.blocks['head'] = nn.Sequential(module, MN2.classifier)

        # self.drop_blocks = ['block3', 'block5', 'block7', 'block8', 'block10', 'block11', 'block13', 'block14',
        #                     'block15']

    def forward(self, x):
        out = x
        for block_name, block in self.blocks.items():
            out = block(out)

        return out


class ReconstructEfficientNet(nn.Module):
    def __init__(self, EfficientNet, infos=None):
        super(ReconstructEfficientNet, self).__init__()
        if infos is None:
            infos = [0 for i in range(9)]
        num_blocks = 16
        self.keep_blocks = ['block1', 'block2', 'block4', 'block6', 'block9', 'block12', 'block16']
        self.drop_blocks = ['block3', 'block5', 'block7', 'block8', 'block10', 'block11', 'block13', 'block14', 'block15']

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = EfficientNet.blocks.base

        for i in range(num_blocks):
            if f'block{i + 1}' in self.keep_blocks:
                self.blocks[f'block{i + 1}'] = getattr(EfficientNet.blocks, f'block{i + 1}')
            elif f'block{i + 1}' in self.drop_blocks and infos[self.drop_blocks.index(f'block{i + 1}')] == 0:
                self.blocks[f'block{i + 1}'] = getattr(EfficientNet.blocks, f'block{i + 1}')

        self.blocks['head'] = EfficientNet.blocks.head

    def forward(self, x):
        out = x
        for block_name, block in self.blocks.items():
            out = block(out)

        return out


class NewPolicyEfficientNet(nn.Module):
    def __init__(self, EN):
        super(NewPolicyEfficientNet, self).__init__()

        self.blocks = EN.blocks
        self.keep_blocks = ['block1', 'block2', 'block4', 'block6', 'block9', 'block12', 'block16']
        self.drop_blocks = ['block3', 'block5', 'block7', 'block8', 'block10', 'block11', 'block13', 'block14',
                            'block15']
        self.num_blocks = 16

    def forward(self, x, policy):
        # (9,2) -- > (9,)
        policy = policy[:, 0]  # reserve
        out = x
        out = self.blocks['base'](out)

        for i in range(self.num_blocks):
            if f'block{i + 1}' in self.keep_blocks:
                out = self.blocks[f'block{i + 1}'](out)
            elif f'block{i + 1}' in self.drop_blocks:
                idx = self.drop_blocks.index(f'block{i + 1}')
                action = policy[idx].contiguous()
                action_mask = action.float().view(-1, 1, 1, 1)
                # version 1
                if action.item() == 1:   #reserve
                    out = self.blocks[f'block{i + 1}'](out) * action_mask
                # version 2
                # out = self.blocks[f'block{i + 1}'](out) * action_mask

        out = self.blocks['head'](out)

        return out


if __name__ == '__main__':
    model = EfficientNet.from_name('efficientnet-b8')
    model = EfficientNet.from_name('efficientnet-b0')

    # cifar100
    model._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=False)
    num_classes = 100  # Example: Set to 10 for 10 output classes
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=num_classes)

    # splitEN = BlockSplitEfficientNet(model)
    # x = torch.ones(10, 3, 32, 32)
    # y = splitEN(x)
    #
    # newEN = ReconstructEfficientNet(splitEN, infos=[1,1,0,0,0,0,0,0,0])
    # x = torch.ones(10, 3, 32, 32)
    # y = newEN(x)

    # 能否正常训练
    drop_blocks = ['block3', 'block5', 'block7', 'block8', 'block10', 'block11', 'block13', 'block14',
                   'block15']
    splitEN = BlockSplitEfficientNet(model)
    batch_model = NewPolicyEfficientNet(splitEN)
    policy_network = policynet(num_classes=9*2)

    optimizer_pn = torch.optim.SGD(policy_network.parameters(), lr=5)
    learning_rate_scheduler_pn = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer_pn,
        gamma=0.1
    )
    loss_acc = nn.CrossEntropyLoss()
    device = "cpu"

    for epoch in range(10):
        x = torch.rand(10, 3, 32, 32)
        y = torch.randint(0, 100, (10,))
        probs = policy_network(x)
        poli = softmax(torch.mean(probs, dim=0).view(-1, 2), dim=1)
        poli, probs = gumbel_softmax(poli, device, tau=0.6, hard=True)
        block_flops = torch.tensor([2, 3, 1, 4, 5, 2, 3, 5, 4])
        mask = get_policy(probs, block_flops, 20)
        policy = (mask - poli).detach() + poli

        output = batch_model(x, policy)

        for param in batch_model.parameters():
            param.requires_grad = True
        for i in range(policy.shape[0]):
            if policy[i, 1] == 1:
                drop = getattr(batch_model.blocks, drop_blocks[i])
                for param in drop.parameters():
                    param.requires_grad = False

        parameters_to_optimize = filter(lambda p: p.requires_grad, batch_model.parameters())
        optimizer_n = torch.optim.SGD(parameters_to_optimize, lr=0.1)
        learning_rate_scheduler_n = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer_n,
            gamma=0.1
        )

        # together
        loss = loss_acc(output, y)
        # loss_t = self.loss_t(time_cost, time_cons)
        # loss = loss_acc + self.alpha * loss_t
        optimizer_pn.zero_grad()
        optimizer_n.zero_grad()
        loss.backward()
        optimizer_n.step()
        optimizer_pn.step()
