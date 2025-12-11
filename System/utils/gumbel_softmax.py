import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

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


def get_mask(mask, probs, block_flops, flops_cons, max_num_blocks):
    sorted_indices = torch.argsort(probs[:,0], descending=True)  #降序排列
    cum_flops = 0

    # 只关注保留的blocks超过cons的情况
    if not max_num_blocks:
        for i in sorted_indices:
            if probs[i,0] >= 0.5:
                if cum_flops + block_flops[i] <= flops_cons:
                    cum_flops += block_flops[i]
                else:
                    mask[i,0] -= 1
                    mask[i,1] += 1

    # 关注所有的blocks
    if max_num_blocks:
        for i in sorted_indices:
            if probs[i,0] >= 0.5:
                if cum_flops + block_flops[i] <= flops_cons:
                    cum_flops += block_flops[i]
                else:
                    mask[i,0] -= 1
                    mask[i,1] += 1
            else:
                if cum_flops + block_flops[i] <= flops_cons:
                    cum_flops += block_flops[i]
                    mask[i, 0] += 1
                    mask[i, 1] -= 1

    return mask


def get_mask_new(poli, prob, block_flops, flops_cons, max_num_blocks, device):
    sorted_indices = torch.argsort(prob[:,0], descending=True)  #降序排列
    cum_flops = 0
    # indices = torch.tensor([1, 0]).to(device)
    probs = prob
    # unit_tensor = torch.tensor([1,1]).to(device)
    mask = torch.tensor([True, False]).to(device)

    # 只关注保留的blocks超过cons的情况
    if not max_num_blocks:
        for i in sorted_indices:
            if prob[i,0] >= 0.5:
                if cum_flops + block_flops[i] <= flops_cons:
                    cum_flops += block_flops[i]
                else:
                    poli[i,0] -= 1
                    poli[i,1] += 1
                    probs[i,:] = prob[i,:].masked_fill(mask, 0.01)

    # 关注所有的blocks
    if max_num_blocks:
        for i in sorted_indices:
            if prob[i,0] >= 0.5:
                if cum_flops + block_flops[i] <= flops_cons:
                    cum_flops += block_flops[i]
                else:
                    poli[i, 0] -= 1
                    poli[i, 1] += 1
                    probs[i,:] = prob[i,:].masked_fill(mask, 0.01)
            else:
                if cum_flops + block_flops[i] <= flops_cons:
                    cum_flops += block_flops[i]
                    poli[i, 0] += 1
                    poli[i, 1] -= 1
                    probs[i,:] = prob[i,:].masked_fill(mask, 0.99)

    return poli, probs


# final
def get_policy(probs, block_flops, flops_cons):
    max_flops = torch.sum(block_flops)
    policy = torch.ones(probs.shape[0], probs.shape[1])
    policy[:, 1] = 0

    if max_flops <= flops_cons:
        return policy
    else:
        sorted_indices = torch.argsort(probs[:, 0], descending=False)  # 升序排列 越靠前保留的prob越小
        for i in sorted_indices:
            if max_flops - block_flops[i] > flops_cons:
                max_flops -= block_flops[i]
                policy[i, :] = torch.tensor([0, 1])
            else:
                policy[i, :] = torch.tensor([0, 1])
                return policy
        return policy


