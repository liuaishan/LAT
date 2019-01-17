
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet18(nn.Module):
    def __init__(self, enable_lat, epsilon, pro_num, batch_size = 64, num_classes=200, if_dropout = False):
        self.in_planes = 64
        block = BasicBlock
        num_blocks = [2,2,2,2]
        if torch.cuda.device_count() > 1:
            batch_size = batch_size // torch.cuda.device_count()
        print('| Resnet 18 for ImageNet')
        super(ResNet18, self).__init__()
        self.register_buffer('x_reg', torch.zeros([batch_size, 3, 224, 224]))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.register_buffer('z0_reg', torch.zeros([batch_size, 64, 112, 112]))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(enable_lat, epsilon, pro_num, batch_size,block, 64, num_blocks[0], stride=1, imgSize=56)
        self.layer2 = self._make_layer(enable_lat, epsilon, pro_num, batch_size,block, 128, num_blocks[1], stride=2, imgSize=28)
        self.layer3 = self._make_layer(enable_lat, epsilon, pro_num, batch_size,block, 256, num_blocks[2], stride=2, imgSize=14)
        self.layer4 = self._make_layer(enable_lat, epsilon, pro_num, batch_size,block, 512, num_blocks[3], stride=2, imgSize=7)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num
        self.if_dropout = if_dropout
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, enable_lat, epsilon, pro_num, batch_size, block, planes, num_blocks, stride, imgSize):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        #print(strides)
        for i, stride in enumerate(strides):
            if i == 0 and stride == 2:
                layers.append(block(enable_lat, epsilon, pro_num, batch_size, self.in_planes, planes, stride, 2*imgSize))
            else:
                layers.append(block(enable_lat, epsilon, pro_num, batch_size, self.in_planes, planes, stride, imgSize))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # batch_norm is True for naive ResNet50
        # x for generating adversarial example
        x.retain_grad()
        self.input = x
        if self.enable_lat:
            self.input.retain_grad()
            # LAT add saved grad to x_reg
            input_add = self.input.add(self.epsilon / self.pro_num * self.x_reg.data)
        else:
            input_add = self.input

        self.z0 = self.conv1(input_add)
        if self.enable_lat:
            self.z0.retain_grad()
            # LAT add saved grad to z0_reg
            z0_add = self.z0.add(self.epsilon / self.pro_num * self.z0_reg.data)
        else:
            z0_add = self.z0
        a0 = self.relu(self.bn1(z0_add))
        a0 = self.maxpool(a0)

        a1 = self.layer1(a0)
        a2 = self.layer2(a1)
        a3 = self.layer3(a2)
        a4 = self.layer4(a3)

        p4 = self.avgpool(a4)
        out = p4.view(p4.size(0), -1)
        if (self.if_dropout):
            out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)

        return out

    def zero_reg(self):
        self.x_reg.data = self.x_reg.data.fill_(0.0)
        self.z0_reg.data = self.z0_reg.data.fill_(0.0)
        num_blocks=[2,2,2,2]
        for i in range(0,4):
            for j in range(0,num_blocks[i]):
                for k in range(0,2):
                    exec("self.layer{}[j].z{}_reg.data = self.layer{}[j].z{}_reg.data.fill_(0.0)".format(i+1,k+1,i+1,k+1))

    def adjust_reg(self, new_batchsize):
        self.x_reg.data = torch.zeros([new_batchsize, 3, 224, 224])
        self.z0_reg.data = torch.zeros([new_batchsize, 3, 112, 112])
        num_blocks=[2,2,2,2]
        for i in range(0,4):
            for j in range(0,num_blocks[i]):
                for k in range(0,2):
                    exec("self.layer{}[j].z{}_reg.data = torch.zeros([new_batchsize, 3, self.layer{}[j].imgSize, self.layer{}[j].imgSize])".format(i+1,k+1,i+1,i+1))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, enable_lat, epsilon, pro_num, batch_size, in_planes, planes, stride, imgSize):
        super(BasicBlock, self).__init__()
        self.imgSize = imgSize
        self.planes = planes
        self.init_imgSize = imgSize
        self.conv1 = conv3x3(in_planes, planes, stride)
        if stride != 1:
            self.imgSize = (self.imgSize-3+2*1)//stride + 1 # calculate imageSize after Convolution
        self.register_buffer('z1_reg', torch.zeros([batch_size, planes, self.imgSize, self.imgSize]))
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)

        self.register_buffer('z2_reg', torch.zeros([batch_size, planes, self.imgSize, self.imgSize]))
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num

    def forward(self, x):

        self.z1 = self.conv1(x)   
        if self.enable_lat:
            self.z1.retain_grad()
            # LAT add saved grad to z1_reg
            z1_add = self.z1.add(self.epsilon / self.pro_num * self.z1_reg.data)
        else:
            z1_add = self.z1
        a1 = self.bn1(z1_add)
        a1 = self.relu(a1)

        self.z2 = self.conv2(a1)   
        if self.enable_lat:
            self.z2.retain_grad()
            # LAT add saved grad to z2_reg
            z2_add = self.z2.add(self.epsilon / self.pro_num * self.z2_reg.data)
        else:
            z2_add = self.z2
        a2 = self.bn2(z2_add)

        # if shortcut = sequential(), then shortcut(x) = x
        a2 += self.shortcut(x)
        out = self.relu(a2)

        return out


if __name__ == '__main__':
    x = torch.randn(5,3,224,224)
    net=ResNet18(enable_lat = True,
                 epsilon = 0.3,
                 pro_num = 5,
                 batch_size = 5,
                 num_classes = 200,
                 if_dropout = False)
    y = net(x)
    print(x.size())
    print(y.size())