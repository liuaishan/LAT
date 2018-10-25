'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: test_LAT.py
@time: 2018/10/24 14:30
@desc:
'''

# TODO
# 1. momentum
# 2. hyperparameters: epsilon and alpha
# 3. before or after activation?\

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision


EPOCH = 1
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD_MNIST = False
CLASS_NUM = 10

train_data = torchvision.datasets.MNIST(
    root='C:\\Users\\Eason\\Desktop\\LAT\\MNIST\\',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

print(train_data.train_data.size())

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='C:\\Users\\Eason\\Desktop\\LAT\\MNIST\\', train=False)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:BATCH_SIZE] / 255.
test_y = test_data.test_labels[:BATCH_SIZE]


class naive_CNN(nn.Module):
    def __init__(self):
        super(naive_CNN, self).__init__()
        self.train(True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
            )
        )
        # LAT: The register for saving and restoring gradients
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

        # layer 1
        self.z1 = self.conv1(x)
        # LAT: enable .grad attribute for non-leaf nodes
        self.z1.retain_grad()
        # LAT: add saved grad
        self.z1_add = self.z1.add(self.z1_reg.data)
        a1 = self.relu(self.z1_add)
        p1 = self.maxpool(a1)

        #  layer 2
        self.z2 = self.conv2(p1)
        self.z2.retain_grad()
        self.z2_add = self.z2.add(self.z2_reg.data)
        a2 = self.relu(self.z2_add)
        p2 = self.maxpool(a2)

        # layer 3
        x3 = p2.view(p2.size(0), -1)
        logits = self.out(x3)
        # last layer. Use LAT or not?
        return logits, x3


cnn = naive_CNN()

print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):

        logits = cnn(b_x)[0]

        loss = loss_func(logits, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # before or after activation??
        # LAT: save grad in backward propagation
        cnn.z1_reg.data = cnn.z1.grad
        cnn.z2_reg.data = cnn.z2.grad

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        # print 10 predictions from test data
        test_output, _ = cnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        Accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))
        print('train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.2f' % Accuracy)

