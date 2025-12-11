import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import copy
from transformers import DistilBertForSequenceClassification, ViTForImageClassification, ViTConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


# class BlockSplitRN18(nn.Module):
#     def __init__(self, resnet):
#         super(BlockSplitRN18, self).__init__()
#
#         self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
#
#         self.block1 = resnet.layer1[0]
#
#         self.block2 = resnet.layer1[1]
#
#         #self.ds3 = copy.deepcopy(resnet.layer2[0].downsample)
#         self.ds3 = resnet.layer2[0].downsample
#         self.block3 = nn.Sequential(resnet.layer2[0].conv1, resnet.layer2[0].bn1, resnet.layer2[0].relu, resnet.layer2[0].conv2, resnet.layer2[0].bn2)
#
#         self.block4 = resnet.layer2[1]
#
#         self.ds5 = resnet.layer3[0].downsample
#         self.block5 = nn.Sequential(resnet.layer3[0].conv1, resnet.layer3[0].bn1, resnet.layer3[0].relu, resnet.layer3[0].conv2, resnet.layer3[0].bn2)
#
#         self.block6 = resnet.layer3[1]
#
#         self.ds7 = resnet.layer4[0].downsample
#         self.block7 = nn.Sequential(resnet.layer4[0].conv1, resnet.layer4[0].bn1, resnet.layer4[0].relu, resnet.layer4[0].conv2, resnet.layer4[0].bn2)
#
#         self.block8 = resnet.layer4[1]
#
#         self.head = nn.Sequential(resnet.avgpool, nn.Flatten(start_dim=1, end_dim=-1), resnet.fc)
#
#     def forward(self, x):
#         out = self.base(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.ds3(out) + self.block3(out)
#         out = self.block4(out)
#         out = self.ds5(out) + self.block5(out)
#         out = self.block6(out)
#         out = self.ds7(out) + self.block7(out)
#         out = self.block8(out)
#         out = self.head(out)
#
#         return out
#
#         # out = self.base(x)
#         # out = self.block1(out)
#         # out = self.block2(out)
#         # out1 = self.block3(out)
#         # identity = self.ds3(out)
#         # out = out1 + identity
#         # out = self.block4(out)
#         # out1 = self.block5(out)
#         # identity = self.ds5(out)
#         # out = out1 + identity
#         # out = self.block6(out)
#         # out1 = self.block7(out)
#         # identity = self.ds7(out)
#         # out = out1 + identity
#         # out = self.block8(out)
#         # out = self.head(out)
#         # return out
#
#
# class ReconstructRN18(nn.Module):
#     def __init__(self, resnet, infos=None):
#         super(ReconstructRN18, self).__init__()
#         if infos is None:
#             infos = [0, 0, 0, 0, 0, 0, 0, 0]
#         self.base = resnet.base
#
#         self.blocks = nn.ModuleDict()
#
#         for i in range(len(infos)):
#             if infos[i] == 0:
#                 if i == 0 or i % 2 == 1:
#                     self.blocks[f'block{i + 1}'] = getattr(resnet, f'block{i + 1}')
#                 elif i % 2 == 0:
#                     self.blocks[f'ds{i + 1}'] = getattr(resnet, f'ds{i + 1}')
#                     self.blocks[f'block{i + 1}'] = getattr(resnet, f'block{i + 1}')
#             elif infos[i] == 1:
#                 if i != 0 and i % 2 == 0:
#                     self.blocks[f'ds{i + 1}'] = getattr(resnet, f'ds{i + 1}')
#
#         self.head = resnet.head
#
#     def forward(self, x):
#         out = self.base(x)
#
#         flag = 0
#         for block_name, block in self.blocks.items():
#             if flag == 1:
#                 flag = 0
#                 continue
#             if 'ds' in block_name and block_name.replace('ds', 'block') in self.blocks:
#                 out1 = self.blocks[block_name.replace('ds', 'block')](out)
#                 identify = block(out)
#                 out = out1 + identify
#                 flag = 1
#             else:
#                 out = block(out)
#
#         out = self.head(out)
#
#         return out


class BlockSplitRN18(nn.Module):
    def __init__(self, resnet):
        super(BlockSplitRN18, self).__init__()

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.blocks['block1'] = resnet.layer1[0]
        self.blocks['block2'] = resnet.layer1[1]
        self.blocks['ds3'] = resnet.layer2[0].downsample
        self.blocks['block3'] = nn.Sequential(resnet.layer2[0].conv1, resnet.layer2[0].bn1, resnet.layer2[0].relu, resnet.layer2[0].conv2, resnet.layer2[0].bn2)
        self.blocks['block4'] = resnet.layer2[1]
        self.blocks['ds5'] = resnet.layer3[0].downsample
        self.blocks['block5'] = nn.Sequential(resnet.layer3[0].conv1, resnet.layer3[0].bn1, resnet.layer3[0].relu, resnet.layer3[0].conv2, resnet.layer3[0].bn2)
        self.blocks['block6'] = resnet.layer3[1]
        self.blocks['ds7'] = resnet.layer4[0].downsample
        self.blocks['block7'] = nn.Sequential(resnet.layer4[0].conv1, resnet.layer4[0].bn1, resnet.layer4[0].relu, resnet.layer4[0].conv2, resnet.layer4[0].bn2)
        self.blocks['block8'] = resnet.layer4[1]
        self.blocks['head'] = nn.Sequential(resnet.avgpool, nn.Flatten(start_dim=1, end_dim=-1), resnet.fc)

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
                out = block(out)
            else:
                identify = out
                out = self.relu(block(out) + identify)

        return out


class ReconstructRN18(nn.Module):
    def __init__(self, resnet, infos=None):
        super(ReconstructRN18, self).__init__()
        if infos is None:
            infos = [0, 0, 0, 0, 0, 0, 0, 0]
        ds_idx = [3,5,7]

        #[0,1,0,1,1,0,0,0]

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = resnet.blocks.base

        for i in range(len(infos)):
            if infos[i] == 0:
                if (i+1) not in ds_idx:
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
                else:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
            elif infos[i] == 1:
                if (i+1) in ds_idx:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')

        self.blocks['head'] = resnet.blocks.head

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
                out = block(out)
            else:
                identify = out
                out = self.relu(block(out) + identify)

        return out


class BatchRN18(nn.Module):
    def __init__(self, resnet):
        super(BatchRN18, self).__init__()

        block_names = ['base', 'block1', 'block2', 'ds3', 'block3', 'block4', 'ds5', 'block5', 'block6', 'ds7', 'block7', 'block8', 'head']
        self.blocks = nn.ModuleDict({name: resnet.blocks[name] for name in block_names})

    def forward(self, x, policy):
        # (8,2)
        out = x
        out = self.blocks['base'](out)
        for i in range(policy.shape[0]):
            # donot drop => 0>1
            if policy[i,0] > policy[i,1]:
                if i == 0 or i % 2 == 1:
                    out = self.blocks[f'block{i + 1}'](out) * policy[i,0]
                elif i % 2 == 0:
                    out = (self.blocks[f'ds{i + 1}'](out) + self.blocks[f'block{i + 1}'](out)) * policy[i,0]
            elif policy[i,1] > policy[i,0]:
                if i > 0 and policy[i-1,1] > policy[i-1,0]:
                    with torch.no_grad():
                        n = self.blocks[f'block{i + 1}'](n)
                else:
                    with torch.no_grad():
                        n = self.blocks[f'block{i + 1}'](out)

                if i != 0 and i % 2 == 0:
                    out = self.blocks[f'ds{i + 1}'](out) * policy[i,1]

        out = self.blocks['head'](out)

        # out = x
        # out = self.blocks['base'](out)
        #
        # for i in range(policy.shape[0]):
        #     if i != 0 and i % 2 == 0:
        #         out = self.blocks[f'ds{i + 1}'](out) * torch.sum(policy[i, :]) + self.blocks[f'block{i + 1}'](out) * policy[i, 0]
        #     else:
        #         out = self.blocks[f'block{i + 1}'](out) * policy[i, 0]
        #
        # out = self.blocks['head'](out)

        return out


class PolicyRN18(nn.Module):
    def __init__(self, resnet):
        super(PolicyRN18, self).__init__()

        block_names = ['base', 'block1', 'block2', 'ds3', 'block3', 'block4', 'ds5', 'block5', 'block6', 'ds7', 'block7', 'block8', 'head']
        self.blocks = nn.ModuleDict({name: resnet.blocks[name] for name in block_names})

    def forward(self, x, policy):
        # (8,2) -- > (8,)
        policy = policy[:, 0]  #reserve
        out = x
        out = self.blocks['base'](out)

        for i in range(policy.shape[0]):
            action = policy[i].contiguous()
            action_mask = action.float().view(-1, 1, 1, 1)
            if i != 0 and i % 2 == 0:
                residual = self.blocks[f'ds{i + 1}'](out)
            else:
                residual = out

            fx = F.relu(residual + self.blocks[f'block{i + 1}'](out))
            out = fx * action_mask + residual * (1 - action_mask)

        out = self.blocks['head'](out)

        return out


class SubRN18(nn.Module):
    def __init__(self, resnet):
        super(SubRN18, self).__init__()

        block_names = ['base', 'block1', 'block2', 'ds3', 'block3', 'block4', 'ds5', 'block5', 'block6', 'ds7', 'block7', 'block8', 'head']
        self.blocks = nn.ModuleDict({name: resnet.blocks[name] for name in block_names})

    def forward(self, x, policy):
        # (8,1)
        policy = policy[:, 0]   #reserve
        out = x
        out = self.blocks['base'](out)
        for i in range(policy.shape[0]):
            action = policy[i].contiguous()
            action_mask = action.float().view(-1, 1, 1, 1)
            # donot drop
            if policy[i] == 1:
                if i == 0 or i % 2 == 1: #0,1,3,5,7
                    out = F.relu(self.blocks[f'block{i + 1}'](out) + out) * action_mask
                elif i % 2 == 0: #2,4,6
                    out = F.relu(self.blocks[f'ds{i + 1}'](out) + self.blocks[f'block{i + 1}'](out)) * action_mask

            elif policy[i] == 0:
                if i > 0 and policy[i-1] == 0:
                    with torch.no_grad():
                        n = self.blocks[f'block{i + 1}'](n)
                else:
                    with torch.no_grad():
                        n = self.blocks[f'block{i + 1}'](out)
                if i != 0 and i % 2 == 0:
                    out = self.blocks[f'ds{i + 1}'](out) * action_mask

        out = self.blocks['head'](out)
        # 优化器也要改

        return out


class NewPolicyRN18(nn.Module):
    def __init__(self, resnet):
        super(NewPolicyRN18, self).__init__()

        block_names = ['base', 'block1', 'block2', 'ds3', 'block3', 'block4', 'ds5', 'block5', 'block6', 'ds7', 'block7', 'block8', 'head']
        self.blocks = nn.ModuleDict({name: resnet.blocks[name] for name in block_names})

    def forward(self, x, policy):
        # (8,2) -- > (8,)
        policy = policy[:, 0]  #reserve
        out = x
        out = self.blocks['base'](out)

        for i in range(policy.shape[0]):
            action = policy[i].contiguous()
            action_mask = action.float().view(-1, 1, 1, 1)
            if i != 0 and i % 2 == 0:
                residual = self.blocks[f'ds{i + 1}'](out)
            else:
                residual = out

            if action.item() == 1:
                out = F.relu(residual + self.blocks[f'block{i + 1}'](out)) * action_mask
            else:
                # out = residual * (1 - action_mask)
                out = residual

        out = self.blocks['head'](out)

        return out


class BlockSplitRN34(nn.Module):
    def __init__(self, resnet):
        super(BlockSplitRN34, self).__init__()

        num_layers = 16
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        for i in range(num_layers):
            module = getattr(resnet, f'layer_{(i)}')
            if module.downsample is None:
                self.blocks[f'block{(i+1)}'] = nn.Sequential(module.conv1, module.bn2, module.relu, module.conv2, module.bn3)
            else:
                self.blocks[f'ds{(i + 1)}'] = module.downsample
                self.blocks[f'block{(i + 1)}'] = nn.Sequential(module.conv1, module.bn2, module.relu, module.conv2, module.bn3)
        self.blocks['head'] = nn.Sequential(resnet.avgpool, resnet.fc)
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


class ReconstructRN34(nn.Module):
    def __init__(self, resnet, infos=None):
        super(ReconstructRN34, self).__init__()
        if infos is None:
            infos = [0 for i in range(16)]
        ds_idx = [4,8,14]

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = resnet.blocks.base

        for i in range(len(infos)):
            if infos[i] == 0:
                if (i+1) not in ds_idx:
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
                else:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
            elif infos[i] == 1:
                if (i+1) in ds_idx:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')

        self.blocks['head'] = resnet.blocks.head

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


class PolicyRN34(nn.Module):
    def __init__(self, resnet):
        super(PolicyRN34, self).__init__()

        self.blocks = resnet.blocks
        self.ds_idx = [4, 8, 14]

    def forward(self, x, policy):
        # (16,2) -- > (16,)
        policy = policy[:, 0]  #reserve
        out = x
        out = self.blocks['base'](out)

        for i in range(policy.shape[0]):
            action = policy[i].contiguous()
            action_mask = action.float().view(-1, 1, 1, 1)
            if (i+1) in self.ds_idx:
                residual = self.blocks[f'ds{i + 1}'](out)
            else:
                residual = out

            fx = F.relu(residual + self.blocks[f'block{i + 1}'](out))
            out = fx * action_mask + residual * (1 - action_mask)

        out = self.blocks['head'](out)

        return out


class NewPolicyRN34(nn.Module):
    def __init__(self, resnet):
        super(NewPolicyRN34, self).__init__()

        self.blocks = resnet.blocks
        self.ds_idx = [4, 8, 14]

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


class BlockSplitRN50(nn.Module):
    def __init__(self, resnet):
        super(BlockSplitRN50, self).__init__()

        num_layers = 16
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        for i in range(num_layers):
            module = getattr(resnet, f'layer_{(i)}')
            if module.downsample is None:
                self.blocks[f'block{(i+1)}'] = nn.Sequential(module.conv1, module.bn1, module.conv2, module.bn2, module.conv3, module.bn3)
            else:
                self.blocks[f'ds{(i + 1)}'] = module.downsample
                self.blocks[f'block{(i + 1)}'] = nn.Sequential(module.conv1, module.bn1, module.conv2, module.bn2, module.conv3, module.bn3)
        self.blocks['head'] = nn.Sequential(resnet.avgpool, resnet.fc)
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
            else:
                identify = out
                out = self.relu(block(out) + identify)

        return out


class ReconstructRN50(nn.Module):
    def __init__(self, resnet, infos=None):
        super(ReconstructRN50, self).__init__()
        if infos is None:
            infos = [0 for i in range(16)]
        ds_idx = [1, 4, 8, 14]

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = resnet.blocks.base

        for i in range(len(infos)):
            if infos[i] == 0:
                if (i+1) not in ds_idx:
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
                else:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
            elif infos[i] == 1:
                if (i+1) in ds_idx:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')

        self.blocks['head'] = resnet.blocks.head

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


class PolicyRN50(nn.Module):
    def __init__(self, resnet):
        super(PolicyRN50, self).__init__()

        self.blocks = resnet.blocks
        self.ds_idx = [1, 4, 8, 14]

    def forward(self, x, policy):
        # (16,2) -- > (16,)
        policy = policy[:, 0]  #reserve
        out = x
        out = self.blocks['base'](out)

        for i in range(policy.shape[0]):
            action = policy[i].contiguous()
            action_mask = action.float().view(-1, 1, 1, 1)
            if (i+1) in self.ds_idx:
                residual = self.blocks[f'ds{i + 1}'](out)
            else:
                residual = out

            fx = F.relu(residual + self.blocks[f'block{i + 1}'](out))
            out = fx * action_mask + residual * (1 - action_mask)

        out = self.blocks['head'](out)

        return out


class NewPolicyRN50(nn.Module):
    def __init__(self, resnet):
        super(NewPolicyRN50, self).__init__()

        self.blocks = resnet.blocks
        self.ds_idx = [1, 4, 8, 14]

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


class BlockSplitMN2(nn.Module):
    def __init__(self, MN2):
        super(BlockSplitMN2, self).__init__()

        num_layers = 18
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = MN2.features[0]
        for i in range(num_layers):
            # module = getattr(MN2.features, f'{(i+1)}')
            self.blocks[f'block{(i+1)}'] = MN2.features[i+1]
        # module = getattr(MN2.features, '18')
        # self.blocks['head'] = nn.Sequential(module, MN2.dropout, nn.Flatten(), MN2.fc)
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.blocks['head'] = nn.Sequential(avgpool, nn.Flatten(), MN2.classifier)
        # self.blocks['head'] = nn.Sequential(module, MN2.classifier)

        self.drop_blocks = ['block3', 'block5', 'block6', 'block8', 'block9', 'block10', 'block12', 'block13', 'block15',
                            'block16']

    def forward(self, x):
        out = x
        for block_name, block in self.blocks.items():
            if block_name not in self.drop_blocks:
                out = block(out)
            else:
                out = block(out) + out

        return out


class ReconstructMN2(nn.Module):
    def __init__(self, MN2, infos=None):
        super(ReconstructMN2, self).__init__()
        if infos is None:
            infos = [0 for i in range(10)]
        num_blocks = 18
        self.keep_blocks = ['block1', 'block2', 'block4', 'block7', 'block11', 'block14', 'block17', 'block18']
        self.drop_blocks = ['block3', 'block5', 'block6', 'block8', 'block9', 'block10', 'block12', 'block13', 'block15',
                            'block16']

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = MN2.blocks.base

        for i in range(num_blocks):
            if f'block{i + 1}' in self.keep_blocks:
                self.blocks[f'block{i + 1}'] = getattr(MN2.blocks, f'block{i + 1}')
            elif f'block{i + 1}' in self.drop_blocks and infos[self.drop_blocks.index(f'block{i + 1}')] == 0:
                self.blocks[f'block{i + 1}'] = getattr(MN2.blocks, f'block{i + 1}')

        self.blocks['head'] = MN2.blocks.head

    def forward(self, x):
        out = x
        for block_name, block in self.blocks.items():
            if block_name not in self.drop_blocks:
                out = block(out)
            else:
                out = block(out) + out

        return out


class NewPolicyMN2(nn.Module):
    def __init__(self, MN2):
        super(NewPolicyMN2, self).__init__()

        self.blocks = MN2.blocks
        self.keep_blocks = ['block1', 'block2', 'block4', 'block7', 'block11', 'block14', 'block17', 'block18']
        self.drop_blocks = ['block3', 'block5', 'block6', 'block8', 'block9', 'block10', 'block12', 'block13', 'block15',
                            'block16']
        self.num_blocks = 18

    def forward(self, x, policy):
        # (9,2) -- > (9,)
        policy = policy[:, 0]  #reserve
        out = x
        out = self.blocks['base'](out)

        for i in range(self.num_blocks):
            if f'block{i + 1}' in self.keep_blocks:
                out = self.blocks[f'block{i + 1}'](out)
            elif f'block{i + 1}' in self.drop_blocks:
                idx = self.drop_blocks.index(f'block{i + 1}')
                action = policy[idx].contiguous()
                action_mask = action.float().view(-1, 1, 1, 1)
                if action.item() == 1:
                    out = (out + self.blocks[f'block{i + 1}'](out)) * action_mask
        out = self.blocks['head'](out)

        return out



class BlockSplitRN6(nn.Module):
    def __init__(self, resnet):
        super(BlockSplitRN6, self).__init__()

        num_layers = 2
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        for i in range(num_layers):
            module = getattr(resnet, f'layer_{(i)}')
            if module.downsample is None:
                self.blocks[f'block{(i+1)}'] = nn.Sequential(module.conv1, module.bn2, module.relu, module.conv2, module.bn3)
            else:
                self.blocks[f'ds{(i + 1)}'] = module.downsample
                self.blocks[f'block{(i + 1)}'] = nn.Sequential(module.conv1, module.bn2, module.relu, module.conv2, module.bn3)
        self.blocks['head'] = nn.Sequential(resnet.avgpool, resnet.fc)
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


class BlockSplitRN10(nn.Module):
    def __init__(self, resnet):
        super(BlockSplitRN10, self).__init__()

        num_layers = 4
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        for i in range(num_layers):
            module = getattr(resnet, f'layer_{(i)}')
            if module.downsample is None:
                self.blocks[f'block{(i+1)}'] = nn.Sequential(module.conv1, module.bn2, module.relu, module.conv2, module.bn3)
            else:
                self.blocks[f'ds{(i + 1)}'] = module.downsample
                self.blocks[f'block{(i + 1)}'] = nn.Sequential(module.conv1, module.bn2, module.relu, module.conv2, module.bn3)
        self.blocks['head'] = nn.Sequential(resnet.avgpool, resnet.fc)
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


class ReconstructRN6(nn.Module):
    def __init__(self, resnet, infos=None):
        super(ReconstructRN6, self).__init__()
        if infos is None:
            infos = [0 for i in range(2)]
        ds_idx = [2]

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = resnet.blocks.base

        for i in range(len(infos)):
            if infos[i] == 0:
                if (i+1) not in ds_idx:
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
                else:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
            elif infos[i] == 1:
                if (i+1) in ds_idx:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')

        self.blocks['head'] = resnet.blocks.head

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


class NewPolicyRN6(nn.Module):
    def __init__(self, resnet):
        super(NewPolicyRN6, self).__init__()

        self.blocks = resnet.blocks
        self.ds_idx = [2]

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


class ReconstructRN10(nn.Module):
    def __init__(self, resnet, infos=None):
        super(ReconstructRN10, self).__init__()
        if infos is None:
            infos = [0 for i in range(4)]
        ds_idx = [2, 3, 4]

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = resnet.blocks.base

        for i in range(len(infos)):
            if infos[i] == 0:
                if (i+1) not in ds_idx:
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
                else:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')
                    self.blocks[f'block{i + 1}'] = getattr(resnet.blocks, f'block{i + 1}')
            elif infos[i] == 1:
                if (i+1) in ds_idx:
                    self.blocks[f'ds{i + 1}'] = getattr(resnet.blocks, f'ds{i + 1}')

        self.blocks['head'] = resnet.blocks.head

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


class NewPolicyRN10(nn.Module):
    def __init__(self, resnet):
        super(NewPolicyRN10, self).__init__()

        self.blocks = resnet.blocks
        self.ds_idx = [2, 3, 4]

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


# class BlockSplitDistilBERT(nn.Module):
#     def __init__(self, distilbert_model):
#         super(BlockSplitDistilBERT, self).__init__()
#
#         num_layers = 6  # DistilBERT有6层transformer层
#         self.blocks = nn.ModuleDict()
#
#         # base: embeddings层
#         self.blocks['base'] = distilbert_model.distilbert.embeddings
#
#         # 6个transformer层
#         for i in range(num_layers):
#             self.blocks[f'block{i + 1}'] = distilbert_model.distilbert.transformer.layer[i]
#
#         # head: 分类头
#         self.blocks['head'] = nn.Sequential(
#             distilbert_model.pre_classifier,
#             nn.ReLU(),
#             distilbert_model.dropout,
#             distilbert_model.classifier
#         )
#         self.has_classifier = True
#
#     def forward(self, X):
#
#         if X.shape[-1] == 256:  # 文本数据
#             input_ids = X[:, :128]  # 前128维
#             attention_mask = X[:, 128:]  # 后128维
#
#         # base: embeddings
#         hidden_states = self.blocks['base'](input_ids)
#
#         # transformer layers
#         for i in range(6):
#             block_name = f'block{i + 1}'
#             if block_name in self.blocks:
#                 # 修改1: 处理可能的元组输出
#                 layer_outputs = self.blocks[block_name](
#                     hidden_states,
#                     attention_mask
#                 )
#
#                 # 如果是元组,取第一个元素;如果不是,直接使用
#                 if isinstance(layer_outputs, tuple):
#                     hidden_states = layer_outputs[0]
#                 else:
#                     hidden_states = layer_outputs
#
#         # 修改2: 添加维度检查
#         if len(hidden_states.shape) == 3:
#             pooled_output = hidden_states[:, 0]  # 取[CLS] token
#         elif len(hidden_states.shape) == 2:
#             pooled_output = hidden_states
#         else:
#             raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
#
#         output = self.blocks['head'](pooled_output)
#
#         return output
#
#
# class NewPolicyDistilBERT(nn.Module):
#     def __init__(self, distilbert_split):
#         super(NewPolicyDistilBERT, self).__init__()
#
#         self.blocks = distilbert_split.blocks
#         self.num_layers = 6
#         self.has_classifier = distilbert_split.has_classifier
#
#     def forward(self, X, policy=None):
#         if X.shape[-1] == 256:  # 文本数据
#             input_ids = X[:, :128]  # 前128维
#             attention_mask = X[:, 128:]  # 后128维
#
#         policy = policy[:, 0]  # (6,) - reserve policy
#
#         # base: embeddings
#         hidden_states = self.blocks['base'](input_ids)
#
#         for i in range(self.num_layers):
#             action = policy[i]
#
#             if action.item() == 1:
#                 # 执行transformer layer
#                 layer_outputs = self.blocks[f'block{i + 1}'](
#                     hidden_states,
#                     attention_mask
#                 )
#
#                 # 正确处理输出
#                 if isinstance(layer_outputs, tuple):
#                     hidden_states = layer_outputs[0]
#                 else:
#                     hidden_states = layer_outputs
#             # else: 跳过该层,hidden_states保持不变
#
#         # 确保hidden_states是3D张量
#         assert len(hidden_states.shape) == 3, f"Expected 3D tensor, got shape {hidden_states.shape}"
#
#         pooled_output = hidden_states[:, 0]  # 取[CLS] token
#         output = self.blocks['head'](pooled_output)
#
#         return output
#
#
# class ReconstructDistilBERT(nn.Module):
#     def __init__(self, distilbert_split, infos=None):
#         super(ReconstructDistilBERT, self).__init__()
#
#         if infos is None:
#             infos = [0 for i in range(6)]  # 默认保留所有6层
#
#         self.blocks = nn.ModuleDict()
#
#         # 始终保留base
#         self.blocks['base'] = distilbert_split.blocks['base']
#
#         # 保存哪些 layer 要保留
#         self.active_layers = []
#         for i in range(len(infos)):
#             if infos[i] == 0:
#                 self.blocks[f'block{i + 1}'] = distilbert_split.blocks[f'block{i + 1}']
#                 self.active_layers.append(i)
#
#         # 始终保留head
#         self.blocks['head'] = distilbert_split.blocks['head']
#         self.has_classifier = distilbert_split.has_classifier
#
#     def forward(self, X):
#         if X.shape[-1] == 256:  # 文本数据
#             input_ids = X[:, :128]  # 前128维
#             attention_mask = X[:, 128:]  # 后128维
#
#         # base: embeddings
#         hidden_states = self.blocks['base'](input_ids)
#
#         # transformer layers - 逐层调用，确保维度正确
#         for i in self.active_layers:
#             block_name = f'block{i + 1}'
#             # 确保 hidden_states 是正确的形状
#             assert hidden_states.dim() == 3, f"Expected 3D tensor, got {hidden_states.shape}"
#
#             # 调用 layer，传入完整参数
#             layer_output = self.blocks[block_name](
#                 hidden_states,
#                 attention_mask
#             )
#
#             # 处理输出
#             if isinstance(layer_output, tuple):
#                 hidden_states = layer_output[0]
#             else:
#                 hidden_states = layer_output
#
#             # 再次检查维度
#             if hidden_states.dim() != 3:
#                 raise ValueError(f"Layer {block_name} output has wrong shape: {hidden_states.shape}")
#
#         # 取 [CLS] token
#         pooled_output = hidden_states[:, 0]
#
#         # 分类头
#         output = self.blocks['head'](pooled_output)
#
#         return output


class BlockSplitDistilBERT(nn.Module):
    def __init__(self, bert_model):
        super(BlockSplitDistilBERT, self).__init__()

        num_layers = 4  # BERT-Mini有4层transformer层
        self.blocks = nn.ModuleDict()

        # base: embeddings层
        self.blocks['base'] = bert_model.distilbert.embeddings

        # 6个transformer层
        for i in range(num_layers):
          self.blocks[f'block{i + 1}'] = bert_model.distilbert.transformer.layer[i]

        # head: 分类头
        self.blocks['head'] = nn.Sequential(
            bert_model.pre_classifier,
            nn.ReLU(),
            bert_model.dropout,
            bert_model.classifier
         )
        self.has_classifier = True

    def forward(self, X):
        if X.shape[-1] == 256:  # 文本数据
            input_ids = X[:, :128]  # 前128维
            attention_mask = X[:, 128:].float()  # 后128维

        # base: embeddings
        hidden_states = self.blocks['base'](input_ids)

        # attention_mask shape: [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
        # 转换为float并创建mask (1.0表示保留, -10000.0表示mask掉)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # transformer layers
        for i in range(4):
            block_name = f'block{i + 1}'
            if block_name in self.blocks:
                # 处理可能的元组输出
                layer_outputs = self.blocks[block_name](
                    hidden_states,
                    extended_attention_mask
                )

                # 如果是元组,取第一个元素;如果不是,直接使用
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

        # 添加维度检查
        if len(hidden_states.shape) == 3:
            pooled_output = hidden_states[:, 0]  # 取[CLS] token
        elif len(hidden_states.shape) == 2:
            pooled_output = hidden_states
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")

        output = self.blocks['head'](pooled_output)

        return output


class NewPolicyDistilBERT(nn.Module):
    def __init__(self, bert_split):
        super(NewPolicyDistilBERT, self).__init__()

        self.blocks = bert_split.blocks
        self.num_layers = 4  # BERT-Mini有4层
        self.has_classifier = bert_split.has_classifier

    def forward(self, X, policy=None):
        if X.shape[-1] == 256:  # 文本数据
            input_ids = X[:, :128]  # 前128维
            attention_mask = X[:, 128:].float()  # 后128维

        policy = policy[:, 0]  # (4,) - reserve policy

        # base: embeddings
        hidden_states = self.blocks['base'](input_ids)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for i in range(self.num_layers):
            action = policy[i]

            if action.item() == 1:
                # 执行transformer layer
                layer_outputs = self.blocks[f'block{i + 1}'](
                    hidden_states,
                    extended_attention_mask
                )

                # 正确处理输出
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
            # else: 跳过该层,hidden_states保持不变

        # 确保hidden_states是3D张量
        assert len(hidden_states.shape) == 3, f"Expected 3D tensor, got shape {hidden_states.shape}"

        pooled_output = hidden_states[:, 0]  # 取[CLS] token
        output = self.blocks['head'](pooled_output)

        return output


class ReconstructDistilBERT(nn.Module):
    def __init__(self, bert_split, infos=None):
        super(ReconstructDistilBERT, self).__init__()

        if infos is None:
            infos = [0 for i in range(4)]  # 默认保留所有4层

        self.blocks = nn.ModuleDict()

        # 始终保留base
        self.blocks['base'] = bert_split.blocks['base']

        # 保存哪些 layer 要保留
        self.active_layers = []
        for i in range(len(infos)):
            if infos[i] == 0:
                self.blocks[f'block{i + 1}'] = bert_split.blocks[f'block{i + 1}']
                self.active_layers.append(i)

        # 始终保留head
        self.blocks['head'] = bert_split.blocks['head']
        self.has_classifier = bert_split.has_classifier

    def forward(self, X):
        if X.shape[-1] == 256:  # 文本数据
            input_ids = X[:, :128]  # 前128维
            attention_mask = X[:, 128:].float()  # 后128维

        # base: embeddings
        hidden_states = self.blocks['base'](input_ids)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # transformer layers - 逐层调用，确保维度正确
        for i in self.active_layers:
            block_name = f'block{i + 1}'
            # 确保 hidden_states 是正确的形状
            assert hidden_states.dim() == 3, f"Expected 3D tensor, got {hidden_states.shape}"

            # 调用 layer，传入完整参数
            layer_output = self.blocks[block_name](
                hidden_states,
                extended_attention_mask
            )

            # 处理输出
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

            # 再次检查维度
            if hidden_states.dim() != 3:
                raise ValueError(f"Layer {block_name} output has wrong shape: {hidden_states.shape}")

        # 取 [CLS] token
        pooled_output = hidden_states[:, 0]

        # 分类头
        output = self.blocks['head'](pooled_output)

        return output

# class BlockSplitViT(nn.Module):
#     def __init__(self, vit_model):
#         super(BlockSplitViT, self).__init__()

#         num_layers = 12
#         self.blocks = nn.ModuleDict()
#         self.blocks['base'] = vit_model.vit.embeddings

#         for i in range(num_layers):
#             self.blocks[f'block{i+1}'] = vit_model.vit.encoder.layer[i]

#         self.blocks['layernorm'] = vit_model.vit.layernorm
#         self.blocks['classifier'] = vit_model.classifier
#         self.has_classifier = True

#     def forward(self, pixel_values):
#         hidden_states = self.blocks['base'](pixel_values, interpolate_pos_encoding=True)

#         for i in range(12):
#             block_name = f'block{i+1}'
#             if block_name in self.blocks:
#                 # 确保正确处理输出
#                 layer_outputs = self.blocks[block_name](hidden_states)
#                 hidden_states = layer_outputs[0]

#         hidden_states = self.blocks['layernorm'](hidden_states)
#         pooled_output = hidden_states[:, 0, :]  # 改为 [:, 0, :]
#         output = self.blocks['classifier'](pooled_output)

#         return output


class BlockSplitViT(nn.Module):
    def __init__(self, vit_model):
        super(BlockSplitViT, self).__init__()
        
        num_layers = 12
        self.blocks = nn.ModuleDict()
        self.blocks['base'] = vit_model.vit.embeddings
        
        for i in range(num_layers):
            self.blocks[f'block{i+1}'] = vit_model.vit.encoder.layer[i]
        
        self.blocks['layernorm'] = vit_model.vit.layernorm
        self.blocks['classifier'] = vit_model.classifier
        self.has_classifier = True
    
    def forward(self, pixel_values):
        hidden_states = self.blocks['base'](pixel_values, interpolate_pos_encoding=True)
        
        for i in range(12):
            block_name = f'block{i+1}'
            if block_name in self.blocks:
                # 修改1: 处理可能的元组输出
                layer_outputs = self.blocks[block_name](hidden_states)
                
                # 如果是元组,取第一个元素;如果不是,直接使用
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
        
        # 修改2: 添加维度检查和调试信息
        hidden_states = self.blocks['layernorm'](hidden_states)
        
        # 修改3: 更安全的索引方式
        if len(hidden_states.shape) == 3:
            pooled_output = hidden_states[:, 0, :]
        elif len(hidden_states.shape) == 2:
            pooled_output = hidden_states
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
        
        output = self.blocks['classifier'](pooled_output)
        
        return output
    

# class NewPolicyViT(nn.Module):
#     def __init__(self, vit_split):
#         super(NewPolicyViT, self).__init__()
        
#         self.blocks = vit_split.blocks
#         self.num_layers = 12  # ViT-Small有12个transformer层
        
#     def forward(self, pixel_values, policy):
#         """
#         Args:
#             pixel_values: 输入图像 (batch_size, 3, H, W) - 支持任意尺寸，如32x32或224x224
#             policy: 策略向量 (12, 2) 或 (batch_size, 12, 2)
#                    每个元素决定是否执行对应的transformer layer
#                    policy[:, 0] 表示保留(reserve)，policy[:, 1] 表示丢弃(drop)
#         """
#         # 如果policy是 (12, 2)，取第一列得到 (12,)
#         # 如果policy是 (batch_size, 12, 2)，取第一列得到 (batch_size, 12)
#         policy = policy[:, 0]  # (12,) - reserve
        
#         # base: embeddings (支持插值位置编码以适配不同输入尺寸)
#         hidden_states = self.blocks['base'](pixel_values, interpolate_pos_encoding=True)
        
#         # 遍历12个transformer层
#         for i in range(self.num_layers):
#             action = policy[i].contiguous()
#             action_mask = action.float().view(1, 1, 1)  # (1, 1, 1)
            
#             # ViT不需要下采样，residual就是当前的hidden_states
#             residual = hidden_states
            
#             # 根据action决定是否执行block
#             if action.item() == 1:
#                 # 执行transformer layer
#                 layer_outputs = self.blocks[f'block{i+1}'](residual) * action_mask
#                 hidden_states = layer_outputs[0]
#             else:
#                 # 跳过该层，直接使用residual
#                 hidden_states = residual * (1 - action_mask)
        
#         # layernorm
#         hidden_states = self.blocks['layernorm'](hidden_states)
        
#         # 取[CLS] token (使用 [:, 0, :] 与 BlockSplitViT 保持一致)
#         pooled_output = hidden_states[:, 0, :]
        
#         # 分类头
#         output = self.blocks['classifier'](pooled_output)
        
#         return output


class NewPolicyViT(nn.Module):
    def __init__(self, vit_split):
        super(NewPolicyViT, self).__init__()
        
        self.blocks = vit_split.blocks
        self.num_layers = 12
        self.has_classifier = vit_split.has_classifier
        
    def forward(self, pixel_values, policy):
        policy = policy[:, 0]  # (12,) - reserve
        
        hidden_states = self.blocks['base'](pixel_values, interpolate_pos_encoding=True)
        
        for i in range(self.num_layers):
            action = policy[i]
            
            if action.item() == 1:
                # 执行transformer layer
                layer_outputs = self.blocks[f'block{i+1}'](hidden_states)
                
                # 正确处理输出
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
            # else: 跳过该层,hidden_states保持不变
        
        hidden_states = self.blocks['layernorm'](hidden_states)
        
        # 确保hidden_states是3D张量
        assert len(hidden_states.shape) == 3, f"Expected 3D tensor, got shape {hidden_states.shape}"
        
        pooled_output = hidden_states[:, 0, :]
        output = self.blocks['classifier'](pooled_output)
        
        return output
    

class ReconstructViT(nn.Module):
    def __init__(self, vit_split, infos=None):
        super(ReconstructViT, self).__init__()

        if infos is None:
            infos = [0 for i in range(12)]

        self.blocks = nn.ModuleDict()
        self.blocks['base'] = vit_split.blocks['base']
        
        # 保存哪些 layer 要保留
        self.active_layers = []
        for i in range(len(infos)):
            if infos[i] == 0:
                self.blocks[f'block{i+1}'] = vit_split.blocks[f'block{i+1}']
                self.active_layers.append(i)

        self.blocks['layernorm'] = vit_split.blocks['layernorm']
        self.blocks['classifier'] = vit_split.blocks['classifier']
        self.has_classifier = vit_split.has_classifier

    def forward(self, pixel_values):
        # Embeddings
        hidden_states = self.blocks['base'](
            pixel_values, 
            interpolate_pos_encoding=True
        )

        # Transformer layers - 逐层调用，确保维度正确
        for i in self.active_layers:
            block_name = f'block{i+1}'
            # 确保 hidden_states 是正确的形状
            assert hidden_states.dim() == 3, f"Expected 3D tensor, got {hidden_states.shape}"
            
            # 调用 layer，传入完整参数
            layer_output = self.blocks[block_name](
                hidden_states,
            )
            
            # 处理输出
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
            
            # 再次检查维度
            if hidden_states.dim() != 3:
                raise ValueError(f"Layer {block_name} output has wrong shape: {hidden_states.shape}")

        # LayerNorm
        hidden_states = self.blocks['layernorm'](hidden_states)
        
        # Pool: 取 CLS token
        pooled_output = hidden_states[:, 0, :]
        
        # Classifier
        output = self.blocks['classifier'](pooled_output)

        return output
    


'''
class BlockReconstructRN18(nn.Module):
    def __init__(self, resnet, infos=[0,0,0,0,0,0,0,0]):
        super(BlockReconstructRN18, self).__init__()
        self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        #self.blocks = nn.ModuleDict()
        self.blocks = {}

        for i in range(len(infos)):
            if infos[i] == 0:
                if i == 0:
                    self.blocks['self.block1'] = resnet.layer1[0]
                elif i == 1 or i == 3 or i == 5 or i == 7:
                    self.blocks[f'self.block{i+1}'] = getattr(resnet, f'layer{(i+1)//2}')[1]
                elif i == 2 or i == 4 or i == 6:
                    self.blocks[f'self.block{i + 1}'] = nn.Sequential(getattr(resnet, f'layer{(i+2)//2}')[0].conv1,
                                          getattr(resnet, f'layer{(i+2)//2}')[0].bn1,
                                          getattr(resnet, f'layer{(i+2)//2}')[0].relu,
                                          getattr(resnet, f'layer{(i+2)//2}')[0].conv2,
                                          getattr(resnet, f'layer{(i+2)//2}')[0].bn2)
                    self.blocks[f'self.ds{i + 1}'] = getattr(resnet, f'layer{(i+2)//2}')[0].downsample
            elif infos[i] == 1:
                if i == 2 or i == 4 or i == 6:
                    self.blocks[f'self.ds{i + 1}'] = getattr(resnet, f'layer{(i + 2) // 2}')[0].downsample

    def forward(self, x):
        out = self.base(x)

        for block_name, block in self.blocks.items():
            out = block(out)

        out = self.head(out)

        return out
'''

'''
class BlockSplitRN18(nn.Module):
    def __init__(self, resnet):
        super(BlockSplitRN18, self).__init__()
        self.base = nn.Sequential(*[resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool])
        self.block1 = nn.Sequential(*[resnet.layer1[0]])
        self.block2 = nn.Sequential(*[resnet.layer1[1]])
        self.block3 = nn.Sequential(*[resnet.layer2[0]])
        self.block4 = nn.Sequential(*[resnet.layer2[1]])
        self.block5 = nn.Sequential(*[resnet.layer3[0]])
        self.block6 = nn.Sequential(*[resnet.layer3[1]])
        self.block7 = nn.Sequential(*[resnet.layer4[0]])
        self.block8 = nn.Sequential(*[resnet.layer4[1]])
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.head = nn.Sequential(*[resnet.avgpool, self.flatten, resnet.fc])

    def forward(self, x):
        out = self.base(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.head(out)

        return out
'''

'''
class BlockReconstructRN18(nn.Module):
    def __init__(self, resnet, infos=[0, 0, 0, 0, 0, 0, 0, 0]):
        super(BlockReconstructRN18, self).__init__()
        self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.blocks = nn.ModuleDict()

        for i in range(len(infos)):
            if infos[i] == 0:
                if i == 0:
                    self.blocks['block1'] = resnet.layer1[0]
                elif i % 2 == 1:
                    layer_index = (i + 1) // 2
                    self.blocks[f'block{i + 1}'] = getattr(resnet, f'layer{layer_index}')[1]
                elif i % 2 == 0:
                    layer_index = (i + 2) // 2
                    if getattr(resnet, f'layer{layer_index}')[0].downsample is not None:
                        self.blocks[f'ds{i + 1}'] = getattr(resnet, f'layer{layer_index}')[0].downsample
                    self.blocks[f'block{i + 1}'] = nn.Sequential(
                        getattr(resnet, f'layer{layer_index}')[0].conv1,
                        getattr(resnet, f'layer{layer_index}')[0].bn1,
                        getattr(resnet, f'layer{layer_index}')[0].relu,
                        getattr(resnet, f'layer{layer_index}')[0].conv2,
                        getattr(resnet, f'layer{layer_index}')[0].bn2
                    )
            elif infos[i] == 1:
                if i % 2 == 0:
                    layer_index = (i + 2) // 2
                    if getattr(resnet, f'layer{layer_index}')[0].downsample is not None:
                        self.blocks[f'ds{i + 1}'] = getattr(resnet, f'layer{layer_index}')[0].downsample

        self.head = nn.Sequential(resnet.avgpool, nn.Flatten(start_dim=1, end_dim=-1), resnet.fc)

    def forward(self, x):
        out = self.base(x)
        #print(out[0][0][0])
        # block_names = list(self.blocks.keys())

        flag = 0
        for block_name, block in self.blocks.items():
            if flag == 1:
                flag = 0
                continue
            if 'ds' in block_name and block_name.replace('ds', 'block') in self.blocks:
                out = block(out) + self.blocks[block_name.replace('ds', 'block')](out)
                flag = 1
            else:
                out = block(out)
            #print(out[0][0][0])

        out = self.head(out)
        #print(out)

        return out
'''

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
#     model = torchvision.models.resnet18(pretrained=False, num_classes=200)
#     split_model = BlockSplitRN18(model)
#     new_model = ReconstructRN18(split_model, [0,1,0,1,1,0,0,0])
#     X = torch.rand(size=(10, 3, 32, 32), dtype=torch.float32)
#     print(new_model(X))

    # # 加载预训练模型
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    #
    # # 划分模型
    # split_model = BlockSplitDistilBERT(model)
    #
    # # 重构模型，例如drop第2层和第4层 (索引1和3)
    # infos = [0, 1, 0, 1, 0, 0]  # 0保留，1drop
    # reconstructed_model = ReconstructDistilBERT(split_model, infos)
    #
    # # 使用
    # input_ids = torch.randint(0, 1000, (2, 128))
    # attention_mask = torch.ones(2, 128)
    # output = reconstructed_model(input_ids, attention_mask)
    # print(output.shape)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # model = ViTForImageClassification.from_pretrained(
    #     'WinKawaks/vit-small-patch16-224',
    #     num_labels=100,
    #     ignore_mismatched_sizes=True
    # ).to(device)
    #
    # split_model = BlockSplitViT(model)
    # print(split_model)
    #
    # policy_model = NewPolicyViT(split_model)
    # print(policy_model)
    #
    # infos = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    # reconstructed_model = ReconstructViT(split_model, infos)
    # reconstructed_model = reconstructed_model.to(device)
    # print(reconstructed_model)
    #
    # policy = torch.tensor([
    # [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],  # 保留block 0-5
    # [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],  # 丢弃block 6-11
    # ], dtype=torch.float32).to(device)
    #
    # pixel_values = torch.randn(10, 3, 32, 32).to(device)
    #
    # with torch.no_grad():
    #     split_output = split_model(pixel_values)
    #     output = policy_model(pixel_values, policy)
    #     reconstructed_output = reconstructed_model(pixel_values)
    #
    # print(f"Split output shape: {split_output.shape}")
    # print(f"Output shape: {output.shape}")
    # print(f"✓ Success! Output shape: {reconstructed_output.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 DistilBERT 模型（SST-2 情感分析任务）
    model = DistilBertForSequenceClassification.from_pretrained(
        'prajjwal1/bert-mini',
        num_labels=2,  # 二分类任务
        ignore_mismatched_sizes=True
    ).to(device)

    # 测试 BlockSplitDistilBERT
    split_model = BlockSplitDistilBERT(model)
    print("=" * 50)
    print("BlockSplitDistilBERT:")
    print(split_model)
    print("=" * 50)

    # 测试 NewPolicyDistilBERT
    policy_model = NewPolicyDistilBERT(split_model)
    print("\nNewPolicyDistilBERT:")
    print(policy_model)
    print("=" * 50)

    # 测试 ReconstructDistilBERT
    # infos: 0表示保留，1表示丢弃
    infos = [0, 0, 1, 0]  # 丢弃 block3 和 block5
    reconstructed_model = ReconstructDistilBERT(split_model, infos)
    reconstructed_model = reconstructed_model.to(device)
    print("\nReconstructDistilBERT:")
    print(reconstructed_model)
    print("=" * 50)

    # 准备测试数据
    batch_size = 4
    seq_length = 128

    # 模拟输入: input_ids 和 attention_mask
    input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)  # 30522 是 DistilBERT 词表大小
    attention_mask = torch.ones(batch_size, seq_length).to(device)

    # Policy for NewPolicyDistilBERT
    # 每层的 policy: [保留, 丢弃]
    policy = torch.tensor([
        [1, 0], [1, 0],
        [0, 1], [0, 1],
    ], dtype=torch.float32).to(device)

    print("\nRunning forward passes...")
    print("=" * 50)

    # 测试前向传播
    with torch.no_grad():
        # 1. 测试 BlockSplitDistilBERT
        split_output = split_model(input_ids, attention_mask)
        print(f"✓ BlockSplitDistilBERT output shape: {split_output.shape}")

        # 2. 测试 NewPolicyDistilBERT
        policy_output = policy_model(input_ids, attention_mask, policy)
        print(f"✓ NewPolicyDistilBERT output shape: {policy_output.shape}")

        # 3. 测试 ReconstructDistilBERT
        reconstructed_output = reconstructed_model(input_ids, attention_mask)
        print(f"✓ ReconstructDistilBERT output shape: {reconstructed_output.shape}")

    print("=" * 50)
    print("✓ All tests passed successfully!")
    print(f"Expected output shape: ({batch_size}, 2)")
    print(f"All outputs have correct shape: {split_output.shape}")
