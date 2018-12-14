
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys

class ResNet50(nn.Module):
    def __init__(self, enable_lat, epsilon, pro_num, batch_size = 32, num_classes=1000, if_dropout = False):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        block = Bottleneck
        num_blocks = [3,4,6,3]
        print('| Resnet 50 for ImageNet')
        self.register_buffer('x_reg', torch.zeros([batch_size, 3, 224, 224]))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.register_buffer('z0_reg', torch.zeros([batch_size, 64, 224, 224]))
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(enable_lat, epsilon, pro_num, batch_size, block, 64, num_blocks[0], stride=1, imgSize=224)
        self.layer2 = self._make_layer(enable_lat, epsilon, pro_num, batch_size, block, 128, num_blocks[1], stride=2, imgSize=112)
        self.layer3 = self._make_layer(enable_lat, epsilon, pro_num, batch_size, block, 256, num_blocks[2], stride=2, imgSize=56)
        self.layer4 = self._make_layer(enable_lat, epsilon, pro_num, batch_size, block, 512, num_blocks[3], stride=2, imgSize=28)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num
        self.if_dropout = if_dropout

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
        a0 = F.relu(self.bn1(z0_add))

        a1 = self.layer1(a0)
        a2 = self.layer2(a1)
        a3 = self.layer3(a2)
        a4 = self.layer4(a3)
        #print(a4.size())
        p4 = F.avg_pool2d(a4,8)
        #print(p4.size())
        out = p4.view(p4.size(0), -1)
        if (self.if_dropout):
            out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear(out)
        
        return out
    
    def zero_reg(self):
        self.x_reg.data = self.x_reg.data.fill_(0.0)
        self.z0_reg.data = self.z0_reg.data.fill_(0.0)
        num_blocks=[3,4,6,3]
        for i in range(0,4):
            for j in range(0,num_blocks[i]):
                for k in range(0,3):
                    exec("self.layer{}[j].z{}_reg.data = self.layer{}[j].z{}_reg.data.fill_(0.0)".format(i+1,k+1,i+1,k+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, enable_lat, epsilon, pro_num, batch_size, in_planes, planes, stride, imgSize):
        super(Bottleneck, self).__init__()
        self.imgSize = imgSize
        self.init_imgSize = imgSize
        self.planes = planes
        #print(self.imgSize)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.register_buffer('z1_reg', torch.zeros([batch_size, planes, self.imgSize, self.imgSize]))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        if stride != 1:
            self.imgSize = round((self.imgSize-3+2*1)/stride) # calculate imageSize after Convolution
        self.register_buffer('z2_reg', torch.zeros([batch_size, planes, self.imgSize, self.imgSize]))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.register_buffer('z3_reg', torch.zeros([batch_size, self.expansion*planes, self.imgSize, self.imgSize]))
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )
        #self.register_buffer('z4_reg', torch.zeros([batch_size, self.expansion*planes, self.imgSize, self.imgSize]))

        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num

    def forward(self, x):
        # batch_norm is True for naive ResNet50 
        # x is the input of block

        self.z1 = self.conv1(x)   
        if self.enable_lat:
            self.z1.retain_grad()
            # LAT add saved grad to z1_reg
            z1_add = self.z1.add(self.epsilon / self.pro_num * self.z1_reg.data)
        else:
            z1_add = self.z1

        a1 = F.relu(self.bn1(z1_add))

        self.z2 = self.conv2(a1)   
        if self.enable_lat:
            self.z2.retain_grad()
            # LAT add saved grad to z2_reg
            z2_add = self.z2.add(self.epsilon / self.pro_num * self.z2_reg.data)
        else:
            z2_add = self.z2
        a2 = F.relu(self.bn2(z2_add))

        self.z3 = self.conv3(a2)   
        if self.enable_lat:
            self.z3.retain_grad()
            # LAT add saved grad to z3_reg
            z3_add = self.z3.add(self.epsilon / self.pro_num * self.z3_reg.data)
        else:
            z3_add = self.z3
        a3 = self.bn3(z3_add)

        z4_sc = self.shortcut(x) + a3
        a4 = F.relu(z4_sc)

        return a4





if __name__ == '__main__':
    net=ResNet50(enable_lat = True,
                 epsilon = 0.3,
                 pro_num = 5,
                 batch_size = 5,
                 num_classes = 10,
                 if_dropout = False)
    x = Variable(torch.randn(5,3,32,32))
    y = net(x)
    print(x.size(),y.size())
