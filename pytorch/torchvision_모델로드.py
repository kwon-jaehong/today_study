from torchvision import models
from torchvision.models import resnet152,ResNet152_Weights
from torchvision.models import resnext101_64x4d,ResNeXt101_64X4D_Weights

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


# model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
model = resnext101_64x4d(weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
resnet_image_extract = nn.Sequential(*list(model.children())[:-2])

input = torch.zeros([1,3,224,224])
# torch.Size([1, 2048, 7, 7])
output = resnet_image_extract(input)

# torch.save(resnet_image_extract.state_dict(), './resnext101_64x4d.pth')
resnet_image_extract.load_state_dict(torch.load('./resnext101_64x4d.pth'))

temp = 1

