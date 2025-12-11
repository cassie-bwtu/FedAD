import torchvision.models as models
import torch.nn as nn
from ptflops import get_model_complexity_info

model = models.densenet201(pretrained=False)  # 不加载预训练模型
model.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

print(model)

flops, params = get_model_complexity_info(model, (3,32,32),as_strings=True,print_per_layer_stat=True)
print("%s |%s" % (flops, params))
