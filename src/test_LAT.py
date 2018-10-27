'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: test_LAT.py
@time: 2018/10/24 14:30
@desc:
'''

# TODO
# 1. before or after activation?

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os


EPOCH = 100
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD_MNIST = True
CLASS_NUM = 10
PRO_NUM = 10 # progress iteration number
EPSILON = 0.1 # noise constraint
ALPHA = 1.0 # velocity of momentum
ENABLE_LAT = True
MODEL_PATH = 'C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\model\\'

train_data = torchvision.datasets.MNIST(
    root='C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\MNIST\\',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)


train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\MNIST\\', train=False)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:BATCH_SIZE] / 255.
test_y = test_data.test_labels[:BATCH_SIZE]


class naive_CNN(nn.Module):
    def __init__(self):
        super(naive_CNN, self).__init__()
        self.train(True)
        # LAT: The register for saving and restoring gradients
        self.register_buffer('x_reg', torch.zeros([BATCH_SIZE, 1, 28, 28]))
        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
            )
        )
        self.register_buffer('z1_reg', torch.zeros([BATCH_SIZE, 16, 28, 28]))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
            )
        )
        # LAT: The register for saving and restoring gradients
        self.register_buffer('z2_reg', torch.zeros([BATCH_SIZE, 32, 14, 14]))
        self.out = nn.Linear(32 * 7 * 7, CLASS_NUM)

    def forward(self, x):

        # layer 0
        self.input = x
        self.input.retain_grad()
        # LAT: add saved grad
        input_add = self.input.add((EPSILON / PRO_NUM) * torch.sign(self.x_reg.data))

        # layer 1
        self.z1 = self.conv1(input_add)
        # LAT: enable .grad attribute for non-leaf nodes
        self.z1.retain_grad()
        z1_add = self.z1.add((EPSILON / PRO_NUM) * torch.sign(self.z1_reg.data))
        a1 = self.relu(z1_add)
        p1 = self.maxpool(a1)

        #  layer 2
        self.z2 = self.conv2(p1)
        self.z2.retain_grad()
        z2_add = self.z2.add((EPSILON / PRO_NUM) * torch.sign(self.z2_reg.data))
        a2 = self.relu(z2_add)
        p2 = self.maxpool(a2)

        # layer 3
        x3 = p2.view(p2.size(0), -1)
        logits = self.out(x3)
        # last layer. Use LAT or not?
        return logits, x3


cnn = naive_CNN()
if ENABLE_LAT:
    if os.path.exists(MODEL_PATH + "lat_param.pkl"):
        cnn.load_state_dict(torch.load(MODEL_PATH + "lat_param.pkl"))
        print('load model.')
    else:
        print("load failed.")
else:
    if os.path.exists(MODEL_PATH + "naive_param.pkl"):
        cnn.load_state_dict(torch.load(MODEL_PATH + "naive_param.pkl"))
        print('load model.')
    else:
        print("load failed.")

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):

        # progressive process
        for iter in range(PRO_NUM):
            iter_input_x = b_x
            iter_input_x.requires_grad = True
            iter_input_x.retain_grad()

            logits = cnn(iter_input_x)[0]
            loss = loss_func(logits, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # before or after activation??
            # LAT: save grad in backward propagation
            # momentum is implemented here
            # L1 norm? or L2 norm?
            cnn.z1_reg.data = ALPHA * cnn.z1_reg.data + \
                              torch.sign(cnn.z1.grad)/torch.norm(cnn.z1.grad, 2)
            cnn.z2_reg.data = ALPHA * cnn.z2_reg.data + \
                              torch.sign(cnn.z2.grad)/torch.norm(cnn.z2.grad, 2)

            cnn.x_reg.data = ALPHA * cnn.x_reg.data + \
                              torch.sign(iter_input_x.grad) / torch.norm(iter_input_x.grad, 2)

            # add or not???? grad of input x
            #temp = torch.clamp(iter_input_x.detach() + EPSILON * torch.sign(iter_input_x.grad),max=1,min=0)
            #iter_input_x = iter_input_x.add(temp)


        # test acc for validation set
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        # save model
        if step % 100 == 0:
            print('saving model...')
            if ENABLE_LAT:
                torch.save(cnn.state_dict(), MODEL_PATH + 'lat_param.pkl')
            else:
                torch.save(cnn.state_dict(), MODEL_PATH + 'naive_param.pkl')


        # print batch-size predictions from test data
        test_output, _ = cnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        Accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))
        print('train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.2f' % Accuracy)



