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
import torch.utils.data as Data
import os
from utils import read_data
from LeNet import LeNet

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
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--test_flag', type = get_bool,default=False, help='test or train')
parser.add_argument('--test_data_path', default=".\\test\\clean.p", help='test dataset path')
parser.add_argument('--val_data_path', default=".\\data\\", help='validation dataset path')
parser.add_argument('--train_data_path', default=".\\adversarial_training_data\\mixed_eps_0.1.p", help='training dataset path')
parser.add_argument('--model_path', default=".\\model\\", help='number of classes')
parser.add_argument('--batchnorm',type = get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type = get_bool,default=True, help='dropout')
parser.add_argument('--dataset', default='mnist', help='data set')
parser.add_argument('--model', default='lenet', help='target model')

args = parser.parse_args()
print(args)


def train_op(model,eps,adv_per):
    #adversarial_training_data(mixed original data with adversarial examples)
    data, label, size = read_data(args.train_data_path)
    
    #check the validity of training_dataset
    if size == 0:
        print("reading data failed.")
        return
	
    # load training data and val set
    data = torch.from_numpy(data).cuda()
    label = torch.from_numpy(label).cuda()
    training_set = Data.TensorDataset(data, label)
    train_loader = Data.DataLoader(dataset=training_set, batch_size=args.batchsize, shuffle=True)
    if args.dataset == 'mnist':
        test_data = torchvision.datasets.MNIST(
            root=args.val_data_path,
            train=False,
			download=True)
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:args.batchsize].cuda() / 255.
        test_y = test_data.test_labels[:args.batchsize].cuda()

	#opt and loss_fuction setting
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    #training procedure
    for epoch in range(args.epoch):
        for step, (x, y) in enumerate(train_loader):
            if not len(y) == args.batchsize:
                continue
            b_x = variable(x).cuda()
            b_y = variable(y).cuda()

            iter_input_x = b_x
            iter_input_x.requires_grad = True
            iter_input_x.retain_grad()

            logits = model(iter_input_x)[0]
            loss = loss_func(logits, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # test acc for validation set
            if step % 50 == 0:
                model.eval()
                test_output, last_layer = model(test_x)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
                model.train()

            # save model
            if step % 100 == 0:
                print('saving model...')
                torch.save(model.state_dict(), args.model_path + 'adv_train_param_eps_%.2f' % eps + '_adv_per_%.2f' % adv_per + '.pkl')

            # print batch-size predictions from training data
            model.eval()
            test_output, _ = model(b_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().numpy().squeeze()
            Accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
            print('train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
            model.train()

def test_op(model):
    # get labels from a .p file
    data, label, size = read_data(args.test_data_path)

    if size == 0:
        print("reading data failed.")
        return
    print(size)
    data = torch.from_numpy(data).cuda()
    label = torch.from_numpy(label).cuda()

    # create dataset
    testing_set = Data.TensorDataset(data, label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=size,
        shuffle=False,
        num_workers=2
    )
    model.eval()
    test_output, _ = model(data)
    pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().numpy().squeeze()
    Accuracy = float((pred_y == label.cpu().numpy()).astype(int).sum()) / float(label.size(0))
    print('test accuracy: %.2f' % Accuracy)
    model.train()


if __name__ == "__main__":
    #eps : size of the perturbation used in training_set
    #adv_per : percentage of adversarial_examples in training_set
    eps = 0.1
    adv_per = 0.5

    #loading the adversarial_training_model
    real_model_path = args.model_path + 'adv_train_param_eps_%.2f' % eps + '_adv_per_%.2f' % adv_per + '.pkl'

    # switch models
    if args.model == 'lenet':
        cnn = LeNet(enable_lat= False,
                    epsilon=0,
                    pro_num=1,
                    batch_size=args.batchsize,
                    batch_norm=args.batchnorm,
                    if_dropout=args.dropout)

    cnn.cuda()

    if os.path.exists(real_model_path):
        cnn.load_state_dict(torch.load(real_model_path))
        print('load model.')
    else:
        print("load failed.")

    if args.test_flag:
        test_op(cnn)
    else:
        train_op(cnn,eps,adv_per)
