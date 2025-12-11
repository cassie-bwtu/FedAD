from torch.nn.functional import gumbel_softmax
# from resnet import *
import torch
# from ptflops import get_model_complexity_info


# images = torch.randn(10,1,28,28)
# policy_network = resnet4(num_classes=8)
# policy_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# probs = policy_network(images)  #(10,8)
#
# action = gumbel_softmax(probs.view(probs.size(0), -1, 2), hard=True)
# policy = action[:,:,1]
#
# print("")
# outputs = net.forward(images, policy)


import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional
from torchvision import models
from flcore.trainmodel.regnet import *
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 用在resnet18中的结构，也就是两个3x3卷积
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        if has_bn:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if has_bn:
            self.bn3 = norm_layer(planes)
        else:
            self.bn3 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
            bn_block_num=4,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_bn:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_bn and (bn_block_num > 0)))
        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1],
                                                has_bn=has_bn and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)

        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_bn=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_bn:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_bn=has_bn))

        return layers

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class PolicyNet(nn.Module):
    def __init__(self, num_classes=8):
        super(PolicyNet, self).__init__()
        self.embedding_layer1 = nn.Linear(1, 4)
        self.embedding_layer2 = nn.Linear(1, 4)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(8, num_classes*2)

    def forward(self, x, y):
        embedded_x = self.embedding_layer1(x)
        embedded_y = self.embedding_layer2(y)
        out = torch.cat([embedded_x, embedded_y], dim=1)  #1,8
        out = self.relu(out)
        out = self.output_layer(out)

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PolicyNetMN2(nn.Module):
    def __init__(self, num_classes=18):
        super(PolicyNetMN2, self).__init__()
        self.base = models.mobilenet_v2(pretrained=False).features[0]
        self.IRBlock = models.mobilenet_v2(pretrained=False).features[1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        last_channel = 16
        self.last_channel = _make_divisible(last_channel * 1.0, 8)
        self.fc = nn.Linear(in_features=self.last_channel, out_features=num_classes)
        # self.out_features = num_classes*2

    def forward(self, x):
        out = self.base(x)
        out = self.IRBlock(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.dropout(out)
        # in_features = out.shape[1]
        # fc = nn.Linear(in_features=in_features, out_features=self.out_features)
        out = self.fc(out)

        return out



class PolicyNetRegNet(nn.Module):
    def __init__(self, num_classes=18):
        super(PolicyNetRegNet, self).__init__()
        regnet = regnet_200m()
        self.base = regnet.stem
        self.block = regnet.s1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.last_channel = 24
        self.fc = nn.Linear(in_features=self.last_channel, out_features=num_classes)
        # self.out_features = num_classes*2

    def forward(self, x):
        out = self.base(x)
        out = self.block(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        # in_features = out.shape[1]
        # fc = nn.Linear(in_features=in_features, out_features=self.out_features)
        out = self.fc(out)

        return out


# def policynet(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
#     return ResNet(BasicBlock, [1], **kwargs)


def policynet(num_blocks: int = 1, num_classes: int = 16, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [num_blocks], num_classes=num_classes, **kwargs)


# policy_network = policynet(num_classes=16)
#
# policy_network_mn2 = PolicyNetMN2()


class PolicyNetwork(nn.Module):
    def __init__(self, num_classes=8):
        super(PolicyNetwork, self).__init__()
        self.fignet = policynet(num_classes=num_classes//2)
        self.embedding_layer1 = nn.Linear(1, num_classes//2)
        self.output_layer = nn.Linear(num_classes, num_classes * 2)

    def forward(self, x, flops):
        out_x = self.fignet(x)
        out_x = torch.mean(out_x, dim=0).view(1, -1)
        out_y = self.embedding_layer1(flops)
        out = torch.cat([out_x, out_y], dim=1)
        out = self.output_layer(out)

        return out


class SimplePolicyNet(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 16):
        super(SimplePolicyNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        # 卷积层 + ReLU 激活
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        # 展平卷积层的输出
        x = x.view(x.size(0), -1)

        # 通过全连接层进行分类
        x = self.fc(x)
        return x

# 创建一个简单的 Policy Network
def policy_netCNN(input_channels: int = 3, num_classes: int = 16) -> SimplePolicyNet:
    return SimplePolicyNet(input_channels=input_channels, num_classes=num_classes)


class LSTMPolicyNet(nn.Module):
    def __init__(
            self,
            vocab_size: int = 30522,
            embed_dim: int = 256,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_classes: int = 12,  # 12个block的drop概率
            bidirectional: bool = True
    ):
        super(LSTMPolicyNet, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        x = self.embedding(input_ids)

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后时刻的hidden state（双向则concat）
        if self.lstm.bidirectional:
            x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            x = h_n[-1]

        # 输出各block的drop概率（logits）
        x = self.fc(x)
        return x


def policy_netLSTM(num_classes: int = 12, **kwargs) -> LSTMPolicyNet:
    """工厂函数创建LSTMPolicyNet"""
    return LSTMPolicyNet(num_classes=num_classes, **kwargs)



# if __name__ == '__main__':
#     policy_network_mn2 = PolicyNetMN2()
#     print(policy_network_mn2)
    # input_tensor = torch.randn(10, 3, 64, 64)
    # output = policy_network_mn2(input_tensor)
    # print(output.shape)

    # flops, params = get_model_complexity_info(policy_network_mn2, (3, 64,64), as_strings=True, print_per_layer_stat=True)
    # print("%s |%s" % (flops, params))

# if __name__ == '__main__':
#     policy_network_regnet = PolicyNetRegNet()
#     print(policy_network_regnet)
#     input_tensor = torch.randn(10, 3, 32, 32)
#     output = policy_network_regnet(input_tensor)
#     print(output.shape)
#
#     flops, params = get_model_complexity_info(policy_network_regnet, (3,32,32), as_strings=True, print_per_layer_stat=True)
#     print("%s |%s" % (flops, params))

# if __name__ == '__main__':
    # pn = policy_netCNN(input_channels= 3, num_classes=16)
    # input_tensor = torch.randn(10, 3, 32, 32)
    # output = pn(input_tensor)
    # print(output.shape)

    # model = policy_netLSTM(num_classes=12)
    # # 模拟输入 (batch_size=4, seq_len=128)
    # input_ids = torch.randint(0, 30522, (4, 128))
    # # 前向传播
    # output = model(input_ids)
    # # 输出形状: (batch_size, 12)
    # print(f"Output shape: {output.shape}")  # torch.Size([4, 12])
