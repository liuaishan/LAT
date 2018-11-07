'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: main.py.py
@time: 2018/10/29 15:36
@desc:
'''

import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
from utils import read_data_label
from LeNet import LeNet
from ResNet import *

from torch.autograd import variable

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

# Training settings
parser = argparse.ArgumentParser(description='lat implementation')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--input_ch', type=int, default=3, help='input image channels')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha')
parser.add_argument('--epsilon', type=float, default=0.6, help='epsilon')
parser.add_argument('--enable_lat', type=get_bool, default=True, help='enable lat')
parser.add_argument('--test_flag', type=get_bool, default=True, help='test or train')
parser.add_argument('--test_data_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\example\\mnist\\fgsm_eps_0.5.p", help='test data path')
parser.add_argument('--test_label_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\example\\mnist\\label.p", help='test label path')
parser.add_argument('--train_data_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\MNIST\\", help='training dataset path')
parser.add_argument('--model_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\model\\new\\", help='number of classes')
parser.add_argument('--pro_num', type=int, default=8, help='progressive number')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--dataset', default='mnist', help='data set')
parser.add_argument('--model', default='lenet', help='target model, [lenet, resnet, vgg, ...]')

args = parser.parse_args()
#print(args)



def train_op(model):

    # load training data and test set
    if args.dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(
            root=args.train_data_path,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=False
        )
        test_data = torchvision.datasets.MNIST(
            root=args.train_data_path,
            train=False)

    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])
        train_data = torchvision.datasets.CIFAR10(
            root=args.train_data_path,
            train=True,
            transform=transform,
            download=False
        )
        test_data = torchvision.datasets.CIFAR10(
            root=args.train_data_path,
            train=False)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)

    if args.dataset == 'mnist':
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:args.batchsize].cuda() / 255.
        test_y = test_data.test_labels[:args.batchsize].cuda()
    if args.dataset == 'cifar10':
        test_x = torch.Tensor(test_data.test_data).view(-1,3,32,32)[:args.batchsize].cuda() / 255.
        test_y = torch.Tensor(test_data.test_labels)[:args.batchsize].cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    curr_lr = args.lr
    for epoch in range(args.epoch):
        for step, (x, y) in enumerate(train_loader):

            if not args.enable_lat:
                args.pro_num = 1
            if not len(y) == args.batchsize:
                continue
            b_x = variable(x).cuda()
            b_y = variable(y).cuda()
            # progressive process
            for iter in range(args.pro_num):
                iter_input_x = b_x
                iter_input_x.requires_grad = True
                iter_input_x.retain_grad()

                logits = model(iter_input_x)
                loss = loss_func(logits, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # before or after activation??
                # LAT: save grad in backward propagation
                # momentum is implemented here
                # L1 norm? or L2 norm?

                if args.model == 'lenet':
                    if args.enable_lat:
                        model.z1_reg.data = args.alpha * model.z1_reg.data + \
                                          torch.sign(model.z1.grad)/torch.norm(model.z1.grad, 2)
                        model.z2_reg.data = args.alpha * model.z2_reg.data + \
                                          torch.sign(model.z2.grad)/torch.norm(model.z2.grad, 2)
                        model.z3_reg.data = args.alpha * model.z3_reg.data + \
                                          torch.sign(model.z3.grad)/torch.norm(model.z3.grad, 2)
                        model.z4_reg.data = args.alpha * model.z4_reg.data + \
                                          torch.sign(model.z4.grad)/torch.norm(model.z4.grad, 2)
                        model.x_reg.data = args.alpha * model.x_reg.data + \
                                            torch.sign(model.input.grad) / torch.norm(model.input.grad, 2)
                        #temp = torch.clamp(iter_input_x.detach() + args.epsilon * torch.sign(iter_input_x.grad),max=1,min=0)

                        #temp = iter_input_x.detach() + args.epsilon * torch.sign(iter_input_x.grad)
                        #iter_input_x = iter_input_x.add(temp)
                        # add or not???? grad of input x
                        # temp = torch.clamp(iter_input_x.detach() + EPSILON * torch.sign(iter_input_x.grad),max=1,min=0)
                        # iter_input_x = iter_input_x.add(temp)
                if args.model == 'resnet':
                    if args.enable_lat:
                        model.z0_reg.data = args.alpha * model.z0_reg.data + \
                                          torch.sign(model.z0.grad)/torch.norm(model.z0.grad, 2)
                        for i in range(0,3):
                            model.layer1[i].z1_reg.data = args.alpha * model.layer1[i].z1_reg.data + \
                                          torch.sign(model.layer1[i].z1.grad)/torch.norm(model.layer1[i].z1.grad, 2)
                            model.layer1[i].z2_reg.data = args.alpha * model.layer1[i].z2_reg.data + \
                                          torch.sign(model.layer1[i].z2.grad)/torch.norm(model.layer1[i].z2.grad, 2)
                            model.layer1[i].z3_reg.data = args.alpha * model.layer1[i].z3_reg.data + \
                                          torch.sign(model.layer1[i].z3.grad)/torch.norm(model.layer1[i].z3.grad, 2)
                            if len(model.layer1[i].shortcut):
                                model.layer1[i].z4_reg.data = args.alpha * model.layer1[i].z4_reg.data + \
                                              torch.sign(model.layer1[i].z4.grad)/torch.norm(model.layer1[i].z4.grad, 2)
                        for j in range(0,4):
                            model.layer2[j].z1_reg.data = args.alpha * model.layer2[j].z1_reg.data + \
                                          torch.sign(model.layer2[j].z1.grad)/torch.norm(model.layer2[j].z1.grad, 2)
                            model.layer2[j].z2_reg.data = args.alpha * model.layer2[j].z2_reg.data + \
                                          torch.sign(model.layer2[j].z2.grad)/torch.norm(model.layer2[j].z2.grad, 2)
                            model.layer2[j].z3_reg.data = args.alpha * model.layer2[j].z3_reg.data + \
                                          torch.sign(model.layer2[j].z3.grad)/torch.norm(model.layer2[j].z3.grad, 2)
                            if len(model.layer2[j].shortcut):
                                model.layer2[j].z4_reg.data = args.alpha * model.layer2[j].z4_reg.data + \
                                              torch.sign(model.layer2[j].z4.grad)/torch.norm(model.layer2[j].z4.grad, 2)
                        for k in range(0,6):
                            model.layer3[k].z1_reg.data = args.alpha * model.layer3[k].z1_reg.data + \
                                          torch.sign(model.layer3[k].z1.grad)/torch.norm(model.layer3[k].z1.grad, 2)
                            model.layer3[k].z2_reg.data = args.alpha * model.layer3[k].z2_reg.data + \
                                          torch.sign(model.layer3[k].z2.grad)/torch.norm(model.layer3[k].z2.grad, 2)
                            model.layer3[k].z3_reg.data = args.alpha * model.layer3[k].z3_reg.data + \
                                          torch.sign(model.layer3[k].z3.grad)/torch.norm(model.layer3[k].z3.grad, 2)
                            if len(model.layer3[k].shortcut):
                                model.layer3[k].z4_reg.data = args.alpha * model.layer3[k].z4_reg.data + \
                                              torch.sign(model.layer3[k].z4.grad)/torch.norm(model.layer3[k].z4.grad, 2)
                        model.x_reg.data = args.alpha * model.x_reg.data + \
                                          torch.sign(model.input.grad)/torch.norm(model.input.grad, 2)


            # test acc for validation set
            if step % 50 == 0:
                model.eval()
                test_output = model(test_x)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy in validation: %.2f' % accuracy)
                model.train()

            # save model
            if step % 100 == 0:
                print('saving model...')
                if args.enable_lat:
                    torch.save(model.state_dict(), args.model_path + 'lat_param.pkl')
                else:
                    torch.save(model.state_dict(), args.model_path + 'naive_param.pkl')

            
            # print batch-size predictions from training data
            model.eval()
            test_output = model(b_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().numpy().squeeze()
            Accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
            print('train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
            model.train()
            
        # Decay learning rate
        if args.model == 'resnet' and (epoch+1) % 20 == 0:
            curr_lr /= 3
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

def test_op(model):
    # get test_data , test_label from .p file
    test_data, test_label, size = read_data_label(args.test_data_path,args.test_label_path)

    if size == 0:
        print("reading data failed.")
        return
    
    #data = torch.from_numpy(data).cuda()
    #label = torch.from_numpy(label).cuda()

    test_data = test_data.cuda()
    test_label = test_label.cuda()
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=size,
        shuffle=False,
        num_workers=2
    )
    model.eval()
    test_output = model(data)
    pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().numpy().squeeze()
    Accuracy = float((pred_y == label.cpu().numpy()).astype(int).sum()) / float(label.size(0))
    print('test accuracy: %.2f' % Accuracy)
    model.train()


if __name__ == "__main__":
    torch.cuda.set_device(4) # use gpu-5
    if args.enable_lat:
        real_model_path = args.model_path + "lat_param.pkl"
        print('loading the LAT model')
    else:
        real_model_path = args.model_path + "naive_param.pkl"
        print('loading the naive model')

    if args.test_flag:
        args.enable_lat = False

    # switch models
    if args.model == 'lenet':
        cnn = LeNet(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    batch_norm=args.batchnorm,
                    if_dropout=args.dropout)
    if args.model == 'resnet':
        cnn = ResNet50(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize)

    cnn.cuda()

    if os.path.exists(real_model_path):
        cnn.load_state_dict(torch.load(real_model_path))
        print('load model.')
    else:
        print("load failed.")

    if args.test_flag:
        test_op(cnn)
    else:
        train_op(cnn)
        
