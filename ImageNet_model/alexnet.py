import torch.nn as nn
from torch.nn import functional as F
import torch


class AlexNet(nn.Module):

    def __init__(self, enable_lat, epsilon, pro_num, batch_size=64, num_classes=1000, if_dropout=False):
        super(AlexNet, self).__init__()
        self.train(True)
        self.if_dropout = if_dropout
        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num

        self.register_buffer('x_reg', torch.zeros([batch_size, 3, 224, 224]))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.register_buffer('z1_reg', torch.zeros([batch_size, 64, 55, 55]))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.register_buffer('z2_reg', torch.zeros([batch_size, 192, 27, 27]))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.register_buffer('z3_reg', torch.zeros([batch_size, 384, 13, 13]))
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.register_buffer('z4_reg', torch.zeros([batch_size, 256, 13, 13]))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.register_buffer('z5_reg', torch.zeros([batch_size, 256, 13, 13]))
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        
    def forward(self,x):
        x.retain_grad()
        self.input = x
        
        if(self.enable_lat):
            self.input.retain_grad()
            input_add = self.input.add(self.epsilon / self.pro_num * self.x_reg.data)
        else:
            input_add = self.input
        
        #layer1
        self.z1 = self.conv1(input_add)
        if(self.enable_lat):
            self.z1.retain_grad()
            z1_add = self.z1.add(self.epsilon / self.pro_num * self.z1_reg.data)
        else:
            z1_add = self.z1
        a1 = F.relu(z1_add)
        p1 = self.maxpool1(a1)
        
        #layer2
        self.z2 = self.conv2(p1)
        if(self.enable_lat):
            self.z2.retain_grad()
            z2_add = self.z2.add(self.epsilon / self.pro_num * self.z2_reg.data)
        else:
            z2_add = self.z2
        a2 = F.relu(z2_add)
        p2 = self.maxpool2(a2) 
        
        #layer3
        self.z3 = self.conv3(p2)
        if(self.enable_lat):
            self.z3.retain_grad()
            z3_add = self.z3.add(self.epsilon / self.pro_num * self.z3_reg.data)
        else:
            z3_add = self.z3
        a3 = F.relu(z3_add)
        
        #layer4
        self.z4 = self.conv4(a3)
        if(self.enable_lat):
            self.z4.retain_grad()
            z4_add = self.z4.add(self.epsilon / self.pro_num * self.z4_reg.data)
        else:
            z4_add = self.z4
        a4 = F.relu(z4_add)
        
        #layer5
        self.z5 = self.conv5(a4)
        if(self.enable_lat):
            self.z5.retain_grad()
            z5_add = self.z5.add(self.epsilon / self.pro_num * self.z5_reg.data)
        else:
            z5_add = self.z5
        a5 = F.relu(z5_add) 
        p3 = self.maxpool3(a5)     

        to_linear = p3.view(p3.size(0),-1)
        if(self.if_dropout):
            output = self.classifier1(to_linear)
        else:
            output = self.classifier2(to_linear)        
        
        return output
        
    def zero_reg(self):
        self.x_reg.data = self.x_reg.data.fill_(0.0)
        self.z1_reg.data = self.z1_reg.data.fill_(0.0)
        self.z2_reg.data = self.z2_reg.data.fill_(0.0)
        self.z3_reg.data = self.z3_reg.data.fill_(0.0)
        self.z4_reg.data = self.z4_reg.data.fill_(0.0)
        self.z5_reg.data = self.z5_reg.data.fill_(0.0)


