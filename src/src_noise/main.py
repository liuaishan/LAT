'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: main.py
@time: 2018/10/29 15:36
@desc:
'''
from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
from utils import *
from LeNet import *
from ResNet import *
from VGG import * 
from denseNet import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from torch.autograd import Variable

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

#device_id = 5
GPUID = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

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
parser.add_argument('--logfile',default='log.txt',help='log file to accord validation process')
parser.add_argument('--enable_noise', type=get_bool, default=False, help='enable gaussian noise on gradient')

'''
parser.add_argument('--loss_path',default='./loss_acc/train_loss/',help='save train loss as .p to draw pic')
parser.add_argument('--acc_path',default='./loss_acc/train_acc/',help='save train acc as .p to draw pic')
'''

args = parser.parse_args()
#print(args)


def train_op(model):
    f=open(args.logfile,'w')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()
    curr_lr = args.lr
    '''
    train_loss = []
    train_acc = []
    Epoch = []
    '''
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch)
        '''
        epoch_loss = []
        epoch_acc = []
        '''
        for step, (x, y) in enumerate(train_loader):

            if not args.enable_lat:
                args.pro_num = 1
            if not len(y) == args.batchsize:
                continue
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            if args.enable_lat:
                model.zero_reg()
            # progressive process
            for iter in range(args.pro_num):
                iter_input_x = b_x
                iter_input_x.requires_grad = True
                iter_input_x.retain_grad()

                logits = model(iter_input_x)
                loss = loss_func(logits, b_y)
                optimizer.zero_grad()
                #------------add gradient noise-------------
                if args.enable_noise:
                    std = 0.3 / (1 + step) ** 0.55
                    loss.register_hook(lambda grad: grad + torch.normal(torch.zeros(grad.size()),std).cuda() ) 
                #-------------------------------------------
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(),args.batchsize)
                optimizer.step()

                # before or after activation??
                # LAT: save grad in backward propagation
                # momentum is implemented here
                # L1 norm? or L2 norm?

                if args.model == 'lenet':
                    if args.enable_lat:
                        model.z1_reg.data = args.alpha * model.z1_reg.data + \
                                          model.z1.grad / torch.norm(torch.norm(torch.norm(model.z1.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,6,28,28)
                        model.z2_reg.data = args.alpha * model.z2_reg.data + \
                                          model.z2.grad / torch.norm(torch.norm(torch.norm(model.z2.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,16,10,10)
                        model.z3_reg.data = args.alpha * model.z3_reg.data + \
                                          model.z3.grad / torch.norm(model.z3.grad, p = 2,dim = 1).view(args.batchsize,1).repeat(1,120)
                        model.z4_reg.data = args.alpha * model.z4_reg.data + \
                                          model.z4.grad / torch.norm(model.z4.grad, p = 2,dim = 1).view(args.batchsize,1).repeat(1,84)
                        model.x_reg.data = args.alpha * model.x_reg.data + \
                                          model.input.grad / torch.norm(torch.norm(torch.norm(model.input.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,1,28,28)

                if args.model == 'resnet':
                    if args.enable_lat:
                        model.z0_reg.data = args.alpha * model.z0_reg.data + \
                                          model.z0.grad / torch.norm(torch.norm(torch.norm(model.z0.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,16,32,32)
                        net = nn.Sequential(model.layer1,model.layer2,model.layer3,model.layer4)
                        #print(len(net),len(net[0]))
                        for i in range(len(net)):
                            for j in range(len(net[i])):
                                net[i][j].z1_reg.data = args.alpha * net[i][j].z1_reg.data + \
                                          net[i][j].z1.grad / torch.norm(torch.norm(torch.norm(net[i][j].z1.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,net[i][j].planes,net[i][j].init_imgSize,net[i][j].init_imgSize) 
                                net[i][j].z2_reg.data = args.alpha * net[i][j].z2_reg.data + \
                                          net[i][j].z2.grad / torch.norm(torch.norm(torch.norm(net[i][j].z2.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,net[i][j].planes,net[i][j].imgSize,net[i][j].imgSize) 
                                net[i][j].z3_reg.data = args.alpha * net[i][j].z3_reg.data + \
                                          net[i][j].z3.grad / torch.norm(torch.norm(torch.norm(net[i][j].z3.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,net[i][j].planes * net[i][j].expansion,net[i][j].imgSize,net[i][j].imgSize) 
                                '''
                                if len(net[i][j].shortcut):
                                    net[i][j].z4_reg.data = args.alpha * net[i][j].z4_reg.data + \
                                              net[i][j].z4.grad / torch.norm(torch.norm(torch.norm(net[i][j].z4.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,net[i][j].planes * net[i][j].expansion,net[i][j].imgSize,net[i][j].imgSize)
                                '''
                        model.x_reg.data = args.alpha * model.x_reg.data + \
                                          model.input.grad / torch.norm(torch.norm(torch.norm(model.input.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,3,32,32)

                if args.model == 'vgg':
                    if args.enable_lat:
                        for i in range(1,13):  # z1 ~ z12
                            exec('model.z{}_reg.data = args.alpha * model.z{}_reg.data + model.z{}.grad / torch.norm(torch.norm(torch.norm(model.z{}.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,model.reg_size_list[{}-1][1],model.reg_size_list[{}-1][2],model.reg_size_list[{}-1][3])'.format(i,i,i,i,i,i,i))
                        model.x_reg.data = args.alpha * model.x_reg.data + \
                                            model.input.grad / torch.norm(torch.norm(torch.norm(model.input.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,3,32,32)

            # test acc for validation set
            if step % 50 == 0:
                if args.enable_lat:
                    model.zero_reg()
                f.write('step {},'.format(step))
                print('epoch={}/{}, step={}/{}'.format(epoch,args.epoch,step,len(train_loader)))
                test_op(model,f)
                if args.enable_lat:
                    test_adv(model,0.031,f)
                    test_adv(model,0.063,f)
            # save model
            if (step+1) % 100 == 0:
                print('saving model...')
                print('lat={}, pro/eps/a={}/{}/{}'.format(args.enable_lat, args.pro_num, args.epsilon, args.alpha))
                if args.enable_lat:
                    torch.save(model.state_dict(), args.model_path + 'lat_param.pkl')
                else:
                    torch.save(model.state_dict(), args.model_path + 'naive_param.pkl')

            
            # print batch-size predictions from training data
            if step % 20 == 0:
                if args.enable_lat:
                    model.zero_reg()
                #model.eval()
                with torch.no_grad():
                    test_output = model(b_x)
                train_loss = loss_func(test_output, b_y)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                Accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
                print('train loss: %.4f' % train_loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
                #model.train()
                
        #end for batch
        '''
        train_loss.append(sum(epoch_loss)/len(epoch_loss))
        train_acc.append(sum(epoch_acc)/len(epoch_acc))
        Epoch.append(epoch)
        '''
    # end for epoch
    '''
    #----------------------save loss and acc-----------------------
    with open(args.loss_acc_path+'train_loss.p','wb') as f:
        pickle.dump(train_loss, f, pickle.HIGHEST_PROTOCOL)
    with open(args.loss_acc_path+'train_acc.p','wb') as f:
        pickle.dump(train_acc, f, pickle.HIGHEST_PROTOCOL)    
    with open(args.loss_acc_path+'epoch.p','wb') as f:
        pickle.dump(Epoch, f, pickle.HIGHEST_PROTOCOL)  
    #--------------------------------------------------------------
    '''
    f.close()


def test_op(model,f=None):
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
        with torch.no_grad():
            h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))        
    if f != None:
        f.write('Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))
        f.write('\n')
    #print('now is {}'.format(type(model)))
    model.train(True)

def test_adv(model,eps,f=None):
    # get test_data , test_label from .p file
    if eps == 0.031:
        data_path = "/media/dsg3/dsgprivate/lat/test/new/test_adv(eps_0.031).p"
    elif eps == 0.063:
        data_path = "/media/dsg3/dsgprivate/lat/test/new/test_adv(eps_0.063).p"
    label_path = "/media/dsg3/dsgprivate/lat/test/new/test_label.p"
    
    test_data, test_label, size = read_data_label(data_path,label_path)

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
        with torch.no_grad():
            h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Accuracy on FGSM eps_{} images: {:.2f} %'.format(eps ,(100 * correct / total)) )        
    if f != None:
        f.write('Accuracy on FGSM eps_{} images: {:.2f} %'.format(eps ,(100 * correct / total)) )
        f.write('\n')
    #print('now is {}'.format(type(model)))
    model.train(True)

if __name__ == "__main__":
    #torch.cuda.set_device(device_id) # use gpu
    if args.enable_lat:
        real_model_path = args.model_path + "lat_param.pkl"
        print('loading the LAT model')
    else:
        real_model_path = args.model_path + "naive_param.pkl"
        print('loading the naive model')
    '''
    if args.test_flag:
        args.enable_lat = False
    '''
    # switch models
    if args.model == 'lenet':
        cnn = LeNet(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    batch_norm=args.batchnorm,
                    if_dropout=args.dropout)
    elif args.model == 'resnet':
        cnn = ResNet50(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
        #cnn.apply(conv_init)
    elif args.model == 'vgg':
        cnn = VGG16(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
    elif args.model == 'densenet':
        cnn = DenseNet()
        
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
        
