'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: main.py
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
from my_utils import read_data_label
from ResNet import *
from VGG import *
from Inception_v2 import *
from math import *
from numpy.random import normal,randint

from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

# Training settings
parser = argparse.ArgumentParser(description='lat implementation')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--input_ch', type=int, default=3, help='input image channels')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--test_flag', type=get_bool, default=True, help='test or train')
parser.add_argument('--test_data_path', default="/media/dsg3/dsgprivate/lat/test/test_data_cln.p", help='test data path')
parser.add_argument('--test_label_path', default="/media/dsg3/dsgprivate/lat/test/tset_label.p", help='test label path')
parser.add_argument('--train_data_path', default="/media/dsg3/dsgprivate/lat/data/cifar10/", help='training dataset path')
parser.add_argument('--model_path', default="/media/dsg3/zhangchongzhi/model/VGG/", help='number of classes')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--model', default='vgg', help='target model, [resnet, vgg, ...]')

args = parser.parse_args()
print(args)

#load the pretrained models
def model_cifar10(model_path_list,model_name_list):
    if(len(model_name_list) != len(model_path_list)):
        print('models\' name and path do not match')
        os._exit(0)
    model_cifar = []
    for index in range(len(model_name_list)):
        if(model_name_list[index] == 'VGG'):
            model = VGG16(enable_lat = False,
                          epsilon = 0,
                          pro_num = 1,
                          batch_size=args.batchsize,
                          if_dropout=args.dropout).cuda()
            if os.path.exists(model_path_list[index]):
                model.load_state_dict(torch.load(model_path_list[index]))
                print('loaded VGG pretrained model.')
            else:
                print("failed to load VGG pretrained model.")
                os._exit(0)
        
        elif(model_name_list[index] == 'ResNet'):
            model = ResNet50(enable_lat = False,
                             epsilon = 0,
                             pro_num = 1,
                             batch_size = args.batchsize,
                             if_dropout = args.dropout).cuda()
            if os.path.exists(model_path_list[index]):
                model.load_state_dict(torch.load(model_path_list[index]))
                print('loaded ResNet pretrained model.')
            else:
                print("failed to load ResNet pretrained model.")
                os._exit(0)
        
        elif(model_name_list[index] == 'Inception_v2'):
            model = Inception_v2().cuda()
            if os.path.exists(model_path_list[index]):
                model.load_state_dict(torch.load(model_path_list[index]))
                print('loaded Inception_v2 pretrained model.')
            else:
                print("failed to load Inception_v2 pretrained model.")
                os._exit(0)
        
        else:
            print('wrong model name')
            os._exit(0)
        
        model_cifar.append(model)
    return model_cifar

def STEP_LL(pretrained_models,training_model,criterion,batch_size,X_input,Y_input):
    '''
    randomly choose the target model from pretrained models and current training model
    '''
    random_number = randint(len(pretrained_models)+1)
    if(random_number == len(pretrained_models)):
        #using current training model 
        model = training_model
    else:
        #using pretrained model
        model = pretrained_models[random_number]

    model.eval()
    eps = []
    for i in range(batch_size):
        while(True):
            rand_num = abs(normal(loc=0.0, scale=8.0, size=None))
            if(rand_num <= 16.0):
                eps.append(float(rand_num)/255.0)
                break

    for  i in range(batch_size):
        X = Variable(X_input[i].clone().expand(1,3,32,32), requires_grad = True).cuda()
        Y = Variable(Y_input[i].clone().expand(1), requires_grad = False).cuda()
        X.retain_grad()
        h= model(X)
        _, predictions = torch.min(h,1)
        loss = criterion(h, predictions)
        model.zero_grad()
        if X.grad is not None:
            X.grad.data.fill_(0)
        loss.backward()
        
        #STEP-LL
        X_adv = X.detach() - eps[i] * torch.sign(X.grad)
        X_adv = torch.clamp(X_adv,0,1)
        
        if(i == 0):
            adv_all_X = X_adv.clone()
            adv_all_Y = Y.clone()
        else:
            adv_all_X = torch.cat((adv_all_X,X_adv),0)
            adv_all_Y = torch.cat((adv_all_Y,Y),0)           
 
    model.train()

    return adv_all_X,adv_all_Y

def train_op(pretrained_models,model):
    # load training data and test set
    transform = transforms.Compose([ transforms.Pad(4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop(32),
                                     transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR10(root = args.train_data_path,
                                              train = True,
                                              transform = transform,
                                              download = False)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        for step, (x, y) in enumerate(train_loader):

            if not len(y) == args.batchsize:
                continue
            
            adv_x,adv_y = STEP_LL(pretrained_models,model,loss_func,args.batchsize,x,y)
            adv_x = Variable(adv_x).cuda()
            adv_y = Variable(adv_y).cuda()
            adv_x.requires_grad = True
            adv_x.retain_grad()
            logits1 = model(adv_x)
            loss1 = loss_func(logits1,adv_y)
            

            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            b_x.requires_grad = True
            b_x.retain_grad()
            logits2 = model(b_x)
            loss2 = loss_func(logits2, b_y)
            
            total_loss = 0.5*(loss1 + loss2)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


            # save model
            if (step+1) % 100 == 0:
                print('epoch : '+ str(epoch))
                print('saving model...')
                torch.save(model.state_dict(), args.model_path + 'ensemble_adv_training_model.pkl')
                test_op(model)

            
            # print batch-size predictions from training data
            if (step+1) % 10 == 0:
                model.eval()
                test_output = model(b_x)
                train_loss = loss_func(test_output, b_y)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                Accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
                print('train loss: %.4f' % train_loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
                model.train(True)
            


def test_op(model):
    # get test_data , test_label from .p file
    test_data, test_label, size = read_data_label(args.test_data_path,args.test_label_path)

    if size == 0:
        print("reading data failed.")
        return


    test_data = test_data.cuda()
    test_label = test_label.cuda()
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batchsize, # without minibatch cuda will out of memory
        shuffle=False,
        #num_workers=2
        drop_last=True,
    )
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    for x, y in testing_loader:
        x = x.cuda()
        y = y.cuda()
        h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))        
    #print('now is {}'.format(type(model)))
    model.train(True)


if __name__ == "__main__":
    model_path_list = ['/media/dsg3/dsgprivate/zhangchongzhi/model/VGG/Inception_v2.pkl',
                       '/media/dsg3/dsgprivate/lat/liuaishan/cifar10/vgg16_origin/naive_param.pkl']
    model_name_list = ['Inception_v2','VGG']
    pretrained_models = model_cifar10(model_path_list,model_name_list)


    real_model_path = args.model_path + 'ensemble_adv_training_model.pkl'
    print('loading the current_training model')

    # switch models
    if args.model == 'resnet':
        net = ResNet50(enable_lat=False,
                       epsilon = 0,
                       pro_num = 1,
                       batch_size=args.batchsize,
                       if_dropout=args.dropout)
        net.apply(conv_init)
    elif args.model == 'vgg':
        net = VGG16(enable_lat = False,
                    epsilon = 0,
                    pro_num = 1,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout)
    else:
        print('the model doesn\'t exist')
        os._exit(0)

    net.cuda()
    
    if os.path.exists(real_model_path):
        net.load_state_dict(torch.load(real_model_path))
        print('load model.')
    else:
        print("load failed.")


    if args.test_flag:
        test_op(net)
    else:
        train_op(pretrained_models,net)
        
