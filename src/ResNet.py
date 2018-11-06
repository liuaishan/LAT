#--------------------------------------------------------------------------------------------------#
# Referenced from https://github.com/meliketoy/wide-resnet.pytorch, by Hang                        #
#--------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys

class ResNet50(nn.Module):
    def __init__(self, enable_lat, epsilon, pro_num, batch_size = 128, num_classes = 10, if_dropout = False):
        super(ResNet50, self).__init__()
        self.in_planes = 16    # input channel 16 
        #self.imageSize = 32    # input dimension 32
        self.batch_size = batch_size
        self.train(True)
        self.if_dropout = if_dropout
        self.block = Bottleneck
        # ResNet-50, block = bottleneck, num_blocks = [3,4,6,3]
        print('| ResNet 50 for CIFAR' )
        self.register_buffer('x_reg', torch.zeros([batch_size, 3, 32, 32]))
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)  # output_size:(batch_size,16,32,32)
        self.bn1 = nn.BatchNorm2d(16)
        self.register_buffer('z0_reg', torch.zeros([batch_size, 16, 32, 32]))
        self.layer1 = nn.Sequential(
            self.block(enable_lat, epsilon, pro_num, batch_size, self.in_planes, 16, 1, 32),
            self.block(enable_lat, epsilon, pro_num, batch_size, 64, 16, 1, 32),
            self.block(enable_lat, epsilon, pro_num, batch_size, 64, 16, 1, 32),
            )   # 3 blocks in layer 1 , plane = 16, imageSize = 32
        self.layer2 = nn.Sequential(
            self.block(enable_lat, epsilon, pro_num, batch_size, 64, 32, 2, 32),   # stride = 2
            self.block(enable_lat, epsilon, pro_num, batch_size, 128, 32, 1, 16),
            self.block(enable_lat, epsilon, pro_num, batch_size, 128, 32, 1, 16),
            self.block(enable_lat, epsilon, pro_num, batch_size, 128, 32, 1, 16),
            ) # 4 blocks in layer 2, plane = 32, imageSize = 32->16,   # what if imageSize turn to 16 ? how to transfer it in brief?
        self.layer3 = nn.Sequential(
            self.block(enable_lat, epsilon, pro_num, batch_size, 128, 64, 2, 16),    # stride = 2
            self.block(enable_lat, epsilon, pro_num, batch_size, 256, 64, 1, 8),
            self.block(enable_lat, epsilon, pro_num, batch_size, 256, 64, 1, 8),
            self.block(enable_lat, epsilon, pro_num, batch_size, 256, 64, 1, 8),
            self.block(enable_lat, epsilon, pro_num, batch_size, 256, 64, 1, 8),
            self.block(enable_lat, epsilon, pro_num, batch_size, 256, 64, 1, 8),
            ) # 6 blocks in layer 3, plane = 64
        '''
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        '''
        self.linear = nn.Linear(64*self.block.expansion, num_classes)

        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num
        
        #defined as class variables, so that instance Block can use these attributes

    '''
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, batch_size))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    '''
    def forward(self, x):
        # x for generating adversarial example
        x.retain_grad()
        self.input = x
        # batch_norm is True for naive ResNet50
        if self.enable_lat:
            self.input.retain_grad()
            # LAT add saved grad to x_reg
            input_add = self.input.add(self.epsilon / self.pro_num * torch.sign(self.x_reg.data))
        else:
            input_add = self.input

        # the original layer:
        # out = F.relu(self.bn1(self.conv1(x)))

        # batch_norm is True for naive ResNet50
        self.z0 = self.bn1(self.conv1(input_add))
        if self.enable_lat:
            self.z0.retain_grad()
            # LAT add saved grad to z0_reg
            z0_add = self.z0.add(self.epsilon / self.pro_num * torch.sign(self.z0_reg.data))
        else:
            z0_add = self.z0
        a0 = F.relu(z0_add)

        #print('layer 1')       # 1*ConvBlock + 2*BasicBlock
        #out = self.layer1(out)
        a1_0 = self.layer1[0](a0  )
        a1_1 = self.layer1[1](a1_0)
        a1_2 = self.layer1[2](a1_1)
        a1 = a1_2
        #print('layer 2')       # 1*ConvBlock + 3*BasicBlock
        #out = self.layer2(out)
        a2_0 = self.layer2[0](a1  )
        a2_1 = self.layer2[1](a2_0)
        a2_2 = self.layer2[2](a2_1)
        a2_3 = self.layer2[3](a2_2)
        a2 = a2_3
        #print('layer 3')       # 1*ConvBlock + 5*BasicBlock
        #out = self.layer3(out)
        a3_0 = self.layer3[0](a2  )
        a3_1 = self.layer3[1](a3_0)
        a3_2 = self.layer3[2](a3_1)
        a3_3 = self.layer3[3](a3_2)
        a3_4 = self.layer3[4](a3_3)
        a3_5 = self.layer3[5](a3_4)
        a3 = a3_5

        #out = F.avg_pool2d(out, 8)
        p3 = F.avg_pool2d(a3, 8)
        #out = out.view(out.size(0), -1)    # change view before FC 
        #out = self.linear(out)
        #print(out.size())
        out = self.linear(p3.view(p3.size(0), -1))
        return out

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

'''
def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': ,
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]
'''
# for ResNet-50, ResNet-101, ResNet-152,
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, enable_lat, epsilon, pro_num, batch_size, in_planes, planes, stride=1, imageSize=32 ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.register_buffer('z1_reg', torch.zeros([batch_size, planes, imageSize, imageSize]))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        #print(imageSize)
        if stride != 1:
            imageSize = round((imageSize-3+2*1)/stride) # calculate imageSize after Convolution
            #print(imageSize, stride)
        self.register_buffer('z2_reg', torch.zeros([batch_size, planes, imageSize, imageSize]))
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.register_buffer('z3_reg', torch.zeros([batch_size, self.expansion*planes, imageSize, imageSize]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
                )
        self.register_buffer('z4_reg', torch.zeros([batch_size, self.expansion*planes, imageSize, imageSize])),

        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num

    def forward(self, x):
        # original conv+bn layer
        # out = F.relu(self.bn1(self.conv1(x)))

        # batch_norm is True for naive ResNet50 
        self.z1 = self.bn1(self.conv1(x))   # here x is the input of block
        if self.enable_lat:
            self.z1.retain_grad()
            # LAT add saved grad to z1_reg
            z1_add = self.z1.add(self.epsilon / self.pro_num * torch.sign(self.z1_reg.data))
        else:
            z1_add = self.z1
        a1 = F.relu(z1_add)
        
        # original conv+bn layer
        # out = F.relu(self.bn2(self.conv2(out)))
        
        # batch_norm is True for naive ResNet50 
        self.z2 = self.bn2(self.conv2(a1))   # here x is the input of block
        if self.enable_lat:
            self.z2.retain_grad()
            # LAT add saved grad to z2_reg
            z2_add = self.z2.add(self.epsilon / self.pro_num * torch.sign(self.z2_reg.data))
        else:
            z2_add = self.z2
        a2 = F.relu(z2_add)

        # original conv+bn layer
        #out = self.bn3(self.conv3(out))

        # batch_norm is True for naive ResNet50 
        self.z3 = self.bn3(self.conv3(a2))   # here x is the input of block
        if self.enable_lat:
            self.z3.retain_grad()
            # LAT add saved grad to z3_reg
            z3_add = self.z3.add(self.epsilon / self.pro_num * torch.sign(self.z3_reg.data))
        else:
            z3_add = self.z3
        a3 = F.relu(z3_add)
        
        # original shortcut layer
        #out += self.shortcut(x)
        if len(self.shortcut) != 0: 
            # shortcut has conv+bn layers
            self.z4 = self.shortcut[1](self.shortcut[0](x))    
            # Dimesion of shortcut(x) = (conv)(bn)*3(x)
            if self.enable_lat:
                self.z4.retain_grad()
                # LAT add saved grad to z4_reg
                z4_add = self.z4.add(self.epsilon / self.pro_num * torch.sign(self.z4_reg.data))
            else:
                z4_add = self.z4    
        else:
            z4_add = 0
        z4_sc = z4_add + a3     # shortcut(x) + (conv)(bn)*3(x) = ConvBlock(x)
        #out = F.relu(out)
        a4 = F.relu(z4_sc)
        
        return a4

'''
# for ResNet-18 and ResNet-34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out
'''
'''
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
'''

if __name__ == '__main__':
    enable_lat = True
    epsilon = 3
    pro_num = 5
    num_classes = 10
    batch_size = 128
    net = ResNet50(enable_lat, epsilon, pro_num, batch_size, num_classes)
    x = torch.randn(batch_size,3,32,32)
    print(x.size())
    y = net(x)   # batchsize = 1, img 3x32x32
    print(y.size())  # output 10 classes 
    '''
    for name, param in net.named_parameters():
        print(name, param.size())
    '''