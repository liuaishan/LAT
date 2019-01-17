'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: main.py
@time: 2018/10/29 15:36
@desc:
'''

import torch
#torch.multiprocessing.set_start_method("spawn")
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
import time
from utils import *
from alexnet import AlexNet
from alexnetBN import AlexNetBN
from resnet import ResNet18
from VGG import * 
from denseNet import *

from torch.autograd import Variable

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

#device_id = 5
GPUID = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

# Training settings
parser = argparse.ArgumentParser(description='lat implementation')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--input_ch', type=int, default=3, help='input image channels')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha')
parser.add_argument('--mu', type=float, default=0.1, help='mu')
parser.add_argument('--epsilon', type=float, default=0.6, help='epsilon')
parser.add_argument('--enable_lat', type=get_bool, default=True, help='enable lat')
parser.add_argument('--test_flag', type=get_bool, default=True, help='test or train')
parser.add_argument('--adv_flag', type=get_bool, default=False, help='adv or clean')
parser.add_argument('--test_data_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\example\\mnist\\fgsm_eps_0.5.p", help='test data path')
parser.add_argument('--test_label_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\example\\mnist\\label.p", help='test label path')
parser.add_argument('--train_data_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\MNIST\\", help='dataset path')
parser.add_argument('--model_path', default="C:\\Users\\SEELE\\Desktop\\LAT\\LAT\\model\\new\\", help='number of classes')
parser.add_argument('--pro_num', type=int, default=8, help='progressive number')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--dataset', default='mnist', help='data set')
parser.add_argument('--model', default='lenet', help='target model, [lenet, resnet, vgg, ...]')
parser.add_argument('--logfile',default='log.txt',help='log file to record validation process')

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

    elif args.dataset == 'cifar10':
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
    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.train_data_path, 'train')
        # ImageFolder: general dataloader
        train_dataset = datasets.ImageFolder(
            traindir,
            # preprocessing
            transforms.Compose([                    
                transforms.RandomResizedCrop(224),      
                transforms.RandomHorizontalFlip(),   
                transforms.ToTensor(),
                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                     std = [ 0.229, 0.224, 0.225 ])
            ]))
        # load train data
        train_loader = Data.DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=True,num_workers=4,drop_last=True)
        valdir = os.path.join(args.train_data_path, 'val')
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # scale up to 256
                transforms.Resize(256),
                # center crop to 224
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                     std = [ 0.229, 0.224, 0.225 ])
            ]))
        val_loader = Data.DataLoader(
                val_dataset,batch_size=args.batchsize, shuffle=False,num_workers=4,drop_last=True)
    if args.dataset != 'imagenet':
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)
    '''
    if args.dataset == 'mnist':
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:args.batchsize].cuda() / 255.
        test_y = test_data.test_labels[:args.batchsize].cuda()
    if args.dataset == 'cifar10':
        test_x = torch.Tensor(test_data.test_data).view(-1,3,32,32)[:args.batchsize].cuda() / 255.
        test_y = torch.Tensor(test_data.test_labels)[:args.batchsize].cuda()
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()
    #curr_lr = args.lr
    cudnn.benchmark = True
    print('data successfully loaded')
    model.train()
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer,epoch)
        end = time.time()
        for step, (x, y) in enumerate(train_loader):
            if step < 5:
                print('step ',step)
            if not args.enable_lat:
                args.pro_num = 1
            if not len(y) == args.batchsize:
                continue
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            if args.enable_lat:
                model.zero_reg()
                #model.module.zero_reg()
            torch.cuda.empty_cache()
            # progressive process
            for iter in range(args.pro_num):
                iter_input_x = b_x
                iter_input_x.requires_grad = True
                iter_input_x.retain_grad()

                logits = model(iter_input_x)
                
                loss = loss_func(logits, b_y)
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(),args.batchsize)
                optimizer.step()

                # before or after activation??
                # LAT: save grad in backward propagation
                # momentum is implemented here
                # L1 norm? or L2 norm?
                if args.model == 'alexnet':
                    if args.enable_lat:
                        model.x_reg.data = args.alpha * model.x_reg.data + \
                                          model.input.grad / torch.norm(torch.norm(torch.norm(model.input.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,3,224,224)
                        model.z1_reg.data = args.alpha * model.z1_reg.data + \
                                          model.z1.grad / torch.norm(torch.norm(torch.norm(model.z1.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,64,55,55)
                        model.z2_reg.data = args.alpha * model.z2_reg.data + \
                                          model.z2.grad / torch.norm(torch.norm(torch.norm(model.z2.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,192,27,27)
                        model.z3_reg.data = args.alpha * model.z3_reg.data + \
                                          model.z3.grad / torch.norm(torch.norm(torch.norm(model.z3.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,384,13,13)
                        model.z4_reg.data = args.alpha * model.z4_reg.data + \
                                          model.z4.grad / torch.norm(torch.norm(torch.norm(model.z4.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,256,13,13)    
                        model.z5_reg.data = args.alpha * model.z5_reg.data + \
                                          model.z5.grad / torch.norm(torch.norm(torch.norm(model.z5.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,256,13,13)
                '''
                if args.model == 'resnet':
                    if args.enable_lat:
                        model.z0_reg.data = args.alpha * model.z0_reg.data + torch.sign(model.z0.grad) 
                        net = nn.Sequential(model.layer1,model.layer2,model.layer3,model.layer4)
                        #print(len(net),len(net[0]))
                        for i in range(len(net)):
                            for j in range(len(net[i])):
                                net[i][j].z1_reg.data = args.alpha * net[i][j].z1_reg.data + torch.sign(net[i][j].z1.grad) 
                                net[i][j].z2_reg.data = args.alpha * net[i][j].z2_reg.data + torch.sign(net[i][j].z2.grad) 
                        model.x_reg.data = args.alpha * model.x_reg.data + torch.sign(model.input.grad) 
                '''
                if args.model == 'resnet':
                    if args.enable_lat:
                        model.z0_reg.data = args.alpha * model.z0_reg.data + \
                                          model.z0.grad / torch.norm(torch.norm(torch.norm(model.z0.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,64,112,112)
                        net = nn.Sequential(model.layer1,model.layer2,model.layer3,model.layer4)
                        #print(len(net),len(net[0]))
                        for i in range(len(net)):
                            for j in range(len(net[i])):
                                net[i][j].z1_reg.data = args.alpha * net[i][j].z1_reg.data + \
                                          net[i][j].z1.grad / torch.norm(torch.norm(torch.norm(net[i][j].z1.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,net[i][j].planes,net[i][j].imgSize,net[i][j].imgSize) 
                                net[i][j].z2_reg.data = args.alpha * net[i][j].z2_reg.data + \
                                          net[i][j].z2.grad / torch.norm(torch.norm(torch.norm(net[i][j].z2.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,net[i][j].planes,net[i][j].imgSize,net[i][j].imgSize) 
                        model.x_reg.data = args.alpha * model.x_reg.data + \
                                          model.input.grad / torch.norm(torch.norm(torch.norm(model.input.grad, p = 2,dim = 2),p = 2,dim = 2),p = 2,dim = 1).view(args.batchsize,1,1,1).repeat(1,3,224,224)
                

            #print('step=',step)
            # print batch-size predictions from training data
            if (step+1) % 100 == 0:
                if args.enable_lat:
                    model.zero_reg()
                    #model.module.zero_reg()
                with torch.no_grad():
                    b_h = model(b_x)
                loss = loss_func(b_h, b_y)
                _, pred_y = torch.max(b_h.data, 1)
                cor = (pred_y == b_y.data).sum().item()
                tot = b_y.size(0)
                acc = (100 * cor/tot)
                f.write('Epoch {}:[{}/{}], Loss:{:.3f}, Train accuracy:{:.2f} %'.format(epoch, step, len(train_loader),loss.item(),acc))
                f.write('\n')
                print('Epoch {}:[{}/{}], Loss:{:.3f}, Train accuracy:{:.2f} %'.format(epoch, step, len(train_loader),loss.item(),acc))
                print('100 step spend {:.3f} time'.format(time.time()-end))
                end = time.time()
            # clear unused cache
            torch.cuda.empty_cache()
            
        #---------------------------------------------------------------------    
        # validate accuracy on validation set (50k)
        if args.enable_lat:
            model.zero_reg()
            #model.module.zero_reg()
        #f.write('step {},'.format(step))
        #print('epoch={}/{}, step={}/{}'.format(epoch,args.epoch,step,len(train_loader)))
        if args.dataset == 'imagenet':
            val_acc = val_op(model,loss_func,val_loader,f)
        if (epoch+1) % 5 == 0:
            test_op(model,f)
        #-----------------------------------------------------------------------    
        # save model
        print('saving model...')
        print('lat={}, pro/eps/a={}/{}/{}'.format(args.enable_lat, args.pro_num, args.epsilon, args.alpha))
        if args.enable_lat:
            torch.save(model.state_dict(), args.model_path + 'lat_param.pkl')
        else:
            torch.save(model.state_dict(), args.model_path + 'naive_param.pkl')
        #-------------------------------------------------------------------------
    f.close()

def val_op(model, criterion, val_loader,f=None):
    torch.cuda.empty_cache()
    val_tot = 0
    val_cor = 0
    # set model to eval
    model.eval()
    for step, (x,y) in enumerate(val_loader):
        print('val_step',step)
        x_val = Variable(x).cuda()
        y_val = Variable(y).cuda()
        with torch.no_grad():
            h = model(x_val)
        loss = criterion(h,y_val)
        _, predicted = torch.max(h.data, 1)
        val_tot += y_val.size(0)
        val_cor += (predicted == y_val.data).sum().item()         
    val_acc = 100 * val_cor / val_tot
    print('Validation accuracy of the model: {:.2f} %'.format(val_acc))
    if f != None:
        f.write('Validation accuracy of the model: {:.2f} %'.format(val_acc))
        f.write('\n')
    model.train()
    return val_acc


def test_op(model,f=None):
    # get test_data , test_label from .p file
    test_data, test_label, size = read_data_label(args.test_data_path,args.test_label_path)

    if size == 0:
        print("reading data failed.")
        return
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
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

    print('Accuracy of the model on the images: {:.2f} %'.format(100 * correct / total))        
    if f != None:
        f.write('Accuracy of the model on the images: {:.2f} %'.format(100 * correct / total))
        f.write('\n')
    #print('now is {}'.format(type(model)))
    model.train(True)

def test_one(model,data_cat,data_path):
    
    label_path = "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/test_label.p"
    test_data, test_label, size = read_data_label(data_path,label_path)

    if size == 0:
        print("reading data failed.")
        return
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
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

    print('Model Acc on {} : {:.2f} %'.format( data_cat,(100 * correct / total)) )        
    model.train(True)

def test_all(model):

    model_list = ['alexnet','resnet18']
    adv_data_alexnet = {
    'fgsm-e8-alexnet':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test/alexnet/test_adv(eps_0.031).p",
    'fgsm-e16-alexnet':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test/alexnet/test_adv(eps_0.063).p",
    'stepll-e8-alexnet':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_stepll/alexnet/test_adv(eps_0.031).p",
    'stepll-e16-alexnet':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_stepll/alexnet/test_adv(eps_0.063).p",
    'pgd-a16-alexnet':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_pgd/alexnet/a_16/test_adv(eps_0.031).p",
    'pgd-a2-alexnet':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_pgd/alexnet/a_2/test_adv(eps_0.031).p",
    'momentum-e8-alexnet':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_momentum/alexnet/test_adv(eps_0.031).p",
    }
    adv_data_resnet18 = {
    'fgsm-e8-resnet18':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/test_adv(eps_0.031).p",
    'fgsm-e16-resnet18':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/test_adv(eps_0.063).p",
    'stepll-e8-resnet18':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_stepll/resnet18/test_adv(eps_0.031).p",
    'stepll-e16-resnet18':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_stepll/resnet18/test_adv(eps_0.063).p",
    'pgd-a16-resnet18':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_pgd/resnet18/a_16/test_adv(eps_0.031).p",
    'pgd-a2-resnet18':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_pgd/resnet18/a_2/test_adv(eps_0.031).p",
    'momentum-e8-resnet18':"/media/dsg3/dsgprivate/lat/sampled_imagenet/test_momentum/resnet18/test_adv(eps_0.031).p",
    }
    
    print('Now is clean data')
    test_op(model)
    
    for target in model_list:
        print('Now adv data come from {}'.format(target))
  
        if target == 'alexnet':
            data_list = adv_data_alexnet
        elif target == 'resnet18':
            data_list = adv_data_resnet18

        for data_cat in data_list:
            data_path = data_list[data_cat]
            test_one(model,data_cat,data_path)


if __name__ == "__main__":

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
    if args.model == 'alexnet':
        cnn = AlexNet(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    num_classes=200,
                    if_dropout=args.dropout
                    )
    elif args.model == 'resnet':
        cnn = ResNet18(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    num_classes=200,
                    if_dropout=args.dropout
                    )
        #cnn.apply(conv_init)
    elif args.model == 'alexnetBN':
        cnn = AlexNetBN(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    num_classes=200,
                    if_dropout=args.dropout
                    )    
    cnn.cuda()

    if os.path.exists(real_model_path):
        cnn.load_state_dict(torch.load(real_model_path))
        print('model successfully loaded.')
    else:
        print("load model failed.")
    
    if args.test_flag:
        if args.adv_flag:
            test_all(cnn)
        else:
            test_op(cnn)
    else:
        train_op(cnn)
        
