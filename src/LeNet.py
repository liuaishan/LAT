'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: LeNet.py.py
@time: 2018/10/29 15:54
@desc:
'''

from torch.autograd import variable
import torch
import torch.nn as nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self, enable_lat, epsilon, pro_num, batch_size=64, class_num = 10, batch_norm=False, if_dropout=False):
        # use the batch_norm after conv layer & use dropout after linear layer
        super(LeNet, self).__init__()
        self.train(True)
        self.batch_norm = batch_norm
        self.if_dropout = if_dropout
        self.bn1 = nn.BatchNorm2d(1)
        self.register_buffer('x_reg', torch.zeros([batch_size, 1, 28, 28]))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                               padding=2)  # output_size:(batch_size,6,28,28)
        self.bn2 = nn.BatchNorm2d(6)
        self.register_buffer('z1_reg', torch.zeros([batch_size, 6, 28, 28]))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # output_size:(batch_size,16,10,10)
        self.bn3 = nn.BatchNorm2d(16)
        self.register_buffer('z2_reg', torch.zeros([batch_size, 16, 10, 10]))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.register_buffer('z3_reg', torch.zeros([batch_size, 120]))
        self.fc2 = nn.Linear(120, 84)
        self.register_buffer('z4_reg', torch.zeros([batch_size, 84]))
        self.fc3 = nn.Linear(84, class_num)

        self.enble_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num

    def forward(self, x):

        # layer 0
        self.input = x
        if (self.batch_norm):
            self.input = self.bn1(self.input)
        if (self.enble_lat):
            self.input.retain_grad()
            # LAT add saved grad to x_reg
            input_add = self.input.add(self.epsilon / self.pro_num * torch.sign(self.x_reg.data))
        else:
            input_add = self.input

        # layer 1
        self.z1 = self.conv1(input_add)
        if (self.batch_norm):
            self.z1 = self.bn2(self.z1)
        if (self.enble_lat):
            self.z1.retain_grad()
            # LAT add saved grad to z1_reg
            z1_add = self.z1.add(self.epsilon / self.pro_num * torch.sign(self.z1_reg.data))
        else:
            z1_add = self.z1
        a1 = F.relu(z1_add)
        p1 = F.max_pool2d(a1, (2, 2))

        # layer 2
        self.z2 = self.conv2(p1)
        if (self.batch_norm):
            self.z2 = self.bn3(self.z2)
        if (self.enble_lat):
            self.z2.retain_grad()
            # LAT add saved grad to z2_reg
            z2_add = self.z2.add(self.epsilon / self.pro_num * torch.sign(self.z2_reg.data))
        else:
            z2_add = self.z2
        a2 = F.relu(z2_add)
        p2 = F.max_pool2d(a2, (2, 2))

        # layer 3
        # do the LAT process before relu function
        self.z3 = self.fc1(p2.view(p2.size(0), -1))
        if (self.if_dropout):
            self.z3 = F.dropout(self.z3, p=0.5, training=self.training)
        if (self.enble_lat):
            self.z3.retain_grad()
            # LAT add saved grad to z3_reg
            z3_add = self.z3.add(self.epsilon / self.pro_num * torch.sign(self.z3_reg.data))
        else:
            z3_add = self.z3
        a3 = F.relu(z3_add)

        # layer 4
        # do the LAT process before relu function
        self.z4 = self.fc2(a3)
        if (self.if_dropout):
            self.z4 = F.dropout(self.z4, p=0.5, training=self.training)
        if (self.enble_lat):
            self.z4.retain_grad()
            # LAT add saved grad to z4_reg
            z4_add = self.z4.add(self.epsilon / self.pro_num * torch.sign(self.z4_reg.data))
        else:
            z4_add = self.z4
        a4 = F.relu(z4_add)

        # layer 5
        logits = self.fc3(a4)

        return logits, a4



