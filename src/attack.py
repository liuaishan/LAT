#--------------------------------------------------------------------
# Reference from https://github.com/1Konny/FGSM , edited by Hang 
#--------------------------------------------------------------------

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np 
import pickle
import os
import argparse


device_id = 3

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True
parser = argparse.ArgumentParser(description='attack implementation')
parser.add_argument('--attack', default='fgsm', help='attack type to be used(fgsm, ifgsm, step_ll, pgd....)')
parser.add_argument('--generate', type=get_bool, default=False, help='whether to generate adv examples as .p files')
# if use iterative attack , the droplast should be set as True
parser.add_argument('--droplast', type=get_bool, default=False, help='whether to drop last batch in testloader')
parser.add_argument('--model', default='resnet', help='target model or model generate advs(resnet, vgg,...)')
parser.add_argument('--modelpath', default="/media/dsg3/dsgprivate/lat/liuaishan/cifar10/vgg16_origin_dropout/naive_param.pkl", help='target model path')
parser.add_argument('--model_batchsize', type=int, default=128, help='batchsize in target model')
parser.add_argument('--dropout', type=get_bool, default=False, help='if dropout in target model')
parser.add_argument('--attack_batchsize', type=int, default=128, help='batchsize in Attack')
parser.add_argument('--attack_epsilon', type=float, default=8.0, help='epsilon in Attack')
parser.add_argument('--attack_alpha', type=float, default=2.0, help='alpha in Attack')
parser.add_argument('--attack_iter', type=int, default=10, help='iteration times in Attack')
parser.add_argument('--attack_momentum', type=float, default=1.0, help='momentum paramter in Attack')
parser.add_argument('--savepath', default="/media/dsg3/dsgprivate/lat/test", help='saving path of clean and adv data')
parser.add_argument('--imgpath', default='/media/dsg3/dsgprivate/lat/img/eps_0.031', help='images path')
parser.add_argument('--enable_lat', type=get_bool, default=False, help='enable lat')
parser.add_argument('--lat_epsilon', type=float, default=0.3, help='epsilon in lat')
parser.add_argument('--lat_pronum', type=int, default=5, help='pronum in lat')
parser.add_argument('--dataset', default='cifar10', help='dataset used for attacking')
args = parser.parse_args()
#print(args)


class Attack():
    def __init__(self, dataroot, dataset, batch_size, target_model, criterion, epsilon=0.2, alpha=0.03, iteration=1):
        self.dataroot = dataroot
        self.dataset = dataset
        self.batch_size = batch_size
        self.model = target_model
        self.criterion = criterion
        self.epsilon = epsilon
        self.alpha = alpha
        self.iteration = iteration
        self.momentum = args.attack_momentum
        
    # root of MNIST/CIFAR-10 testset
    def return_data(self):
        if self.dataset == 'mnist':
            test_dataset = torchvision.datasets.MNIST(root=self.dataroot,train=False, transform=transforms.ToTensor())
        elif self.dataset == 'cifar10':
            test_dataset = torchvision.datasets.CIFAR10(root=self.dataroot,train=False, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=False,drop_last=args.droplast)
        return test_loader
        
    def return_trainloader(self):
        if self.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root=self.dataroot,train=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True)
        return train_loader
        
    def fgsm(self):
        data_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0 
        for i, (images, labels) in enumerate(data_loader):
            x = Variable(images, requires_grad = True).cuda()
            y_true = Variable(labels, requires_grad = False).cuda()
            #print('self.batch_size={},last batch={}'.format(self.batch_size,10000%self.batch_size))
            #if (i+1) == len(test_loader):
            #    self.model.batch_size = 10000 % self.batch_size
            x.retain_grad()
            h = self.model(x)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()
            loss = self.criterion(h, y_true)
            self.model.zero_grad()
            if x.grad is not None:
                x.grad.data.fill_(0)
            loss.backward()
            
            #FGSM
            #x.grad.sign_()   # change the grad with sign ?
            x_adv = x.detach() + self.epsilon * torch.sign(x.grad)
            x_adv = torch.clamp(x_adv,0,1)
            
            h_adv = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()
            print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions_adv == predictions).sum()
            total += len(predictions)
        
        self.model.train()
        
        error_rate = float(total-correct)*100/total
        print("Error Rate is ", float(total-correct)*100/total)
        print("Before FGSM the accuracy is", float(100*correct_cln)/total)
        print("After FGSM the accuracy is", float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv
# MNIST: test_data_cln , torch.Size([10000, 1, 28, 28]) ; test_label, torch.Size([10000])

    def i_fgsm(self):
        test_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0
        for i,(images,labels) in enumerate(test_loader):
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()
            x_adv = Variable(x.data, requires_grad=True).cuda()
            x_adv.retain_grad()
            #if (i+1) == len(test_loader):
            #    self.model.batch_size = 10000 % self.batch_size
            h = self.model(x)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()

            for j in range(0, self.iteration):
                h_adv = self.model(x_adv)

                loss = self.criterion(h_adv, y_true)
                self.model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                loss.backward()
                
                #I-FGSM
                #x_adv.grad.sign_()   # change the grad with sign ?
                print(type(x_adv.grad),x_adv.grad.size())
                x_adv = x_adv.detach() + self.alpha * torch.sign(x_adv.grad)
                # according to the paper of Kurakin:
                x_adv = torch.where(x_adv > x+self.epsilon, x+self.epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = torch.where(x_adv < x-self.epsilon, x-self.epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = Variable(x_adv.data, requires_grad=True).cuda()

            h_adv = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()

            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions == predictions_adv).sum()
            total += len(predictions)
        
        self.model.train()
        error_rate = float(total-correct)*100/total
        print("Error Rate is ",float(total-correct)*100/total)
        print("Before I-FGSM the accuracy is",float(100*correct_cln)/total)
        print("After I-FGSM the accuracy is",float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv 

    def PGD(self):
        test_loader = self.return_data()
        self.model.eval()
        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0
        for i,(images,labels) in enumerate(test_loader):
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()
            h = self.model(x)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()
            x_rand = x.detach()
            x_rand = x_rand + torch.zeros_like(x_rand).uniform_(-self.epsilon,self.epsilon)
            x_rand = torch.clamp(x_rand,0,1)
            x_adv = Variable(x_rand.data, requires_grad=True).cuda()
            for j in range(0, self.iteration):
                print('batch = {}, iter = {}'.format(i,j))
                h_adv = self.model(x_adv)
                loss = self.criterion(h_adv, y_true)
                self.model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                loss.backward()
                
                #I-FGSM
                #x_adv.grad.sign_()   # change the grad with sign ?
                #print(type(x_adv.grad),x_adv.grad.size())
                x_adv = x_adv.detach() + self.alpha * torch.sign(x_adv.grad)
                # according to the paper of Kurakin:
                x_adv = torch.where(x_adv > x+self.epsilon, x+self.epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = torch.where(x_adv < x-self.epsilon, x-self.epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = Variable(x_adv.data, requires_grad=True).cuda()
            h_adv = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()
            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)
            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())
            correct += (predictions == predictions_adv).sum()
            total += len(predictions)
        
        self.model.train()
        error_rate = float(total-correct)*100/total
        print("Error Rate is ",float(total-correct)*100/total)
        print("Before PGD the accuracy is",float(100*correct_cln)/total)
        print("After PGD the accuracy is",float(100*correct_adv)/total)
        return test_data_cln, test_data_adv, test_label, test_label_adv 
        
    def step_ll(self):
        data_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0 
        for i, (images, labels) in enumerate(data_loader):
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()

            h = self.model(x)
            _, predictions = torch.max(h,1)
            _, predictions_ll = torch.min(h,1)
            correct_cln += (predictions == y_true).sum()
            loss = self.criterion(h, predictions_ll)
            self.model.zero_grad()
            if x.grad is not None:
                x.grad.data.fill_(0)
            loss.backward()
            
            # FGSM
            #x.grad.sign_()   # change the grad with sign ?
            x_adv = x.detach() - self.epsilon * torch.sign(x.grad)
            x_adv = torch.clamp(x_adv,0,1)
            
            h_adv = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()
            print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions_adv == predictions).sum()
            total += len(predictions)
        
        self.model.train()
        
        error_rate = float(total-correct)*100/total
        print("Error Rate is ", float(total-correct)*100/total)
        print("Before Step-ll the accuracy is", float(100*correct_cln)/total)
        print("After Step-ll the accuracy is", float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv
        
    def momentum_ifgsm(self):
        test_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0
        for i,(images,labels) in enumerate(test_loader):
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()
            x_adv = Variable(x.data, requires_grad=True).cuda()
            x_grad = torch.zeros(x.size()).cuda()
            #if (i+1) == len(test_loader):
            #    self.model.batch_size = 10000 % self.batch_size
            h = self.model(x)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()

            for j in range(0, self.iteration):
                print('batch = {}, iter = {}'.format(i,j))
                h_adv = self.model(x_adv)

                loss = self.criterion(h_adv, y_true)
                self.model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                loss.backward()
                
                #I-FGSM
                #x_adv.grad.sign_()   # change the grad with sign ?
                #print(type(x_adv.grad),x_adv.grad.size())

                # in Boosting attack
                # alpha = epsilon / iteration

                # calc |x|p except dim=0
                norm = x_adv.grad
                for k in range(1,4):
                    norm = torch.norm(norm,p=1,dim=k).unsqueeze(dim=k)
                # Momentum on gradient noise
                noise = self.momentum * x_grad + x_adv.grad / norm
                x_grad = noise

                x_adv = x_adv.detach() + self.alpha * torch.sign(noise)
                # according to the paper of Kurakin:
                x_adv = torch.where(x_adv > x+self.epsilon, x+self.epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = torch.where(x_adv < x-self.epsilon, x-self.epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = Variable(x_adv.data, requires_grad=True).cuda()

            h_adv = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()

            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions == predictions_adv).sum()
            total += len(predictions)
        
        self.model.train()
        error_rate = float(total-correct)*100/total
        print("Error Rate is ",float(total-correct)*100/total)
        print("Before Momentum IFGSM the accuracy is",float(100*correct_cln)/total)
        print("After Momentum IFGSM the accuracy is",float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv 
        

def save_img(imgpath,test_data_cln, test_data_adv, test_label, test_label_adv):
    #save adversarial example
    if Path(imgpath).exists()==False:
        Path(imgpath).mkdir(parents=True)
    toImg = transforms.ToPILImage()
    image = test_data_cln.cpu()
    image_adv = test_data_adv.cpu()
    label = test_label.cpu()
    label_adv = test_label_adv.cpu()
    tot = len(image)
    batch = 10
    for i in range(0, batch):
        #print(image[i].size())
        im = toImg(image[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_cln.jpg'.format(i,test_label[i])))
        im = toImg(image_adv[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_adv.jpg'.format(i,test_label_adv[i])))

def display(test_data_cln, test_data_adv, test_label, test_label_adv):
    # display a batch adv
    toPil = transforms.ToPILImage()
    curr = test_data_cln.cpu()
    curr_adv = test_data_adv.cpu()
    label = test_label.cpu()
    label_adv = test_label_adv.cpu()
    disp_batch = 10
    for a in range(disp_batch):
        b = a + disp_batch 
        plt.figure()
        plt.subplot(121)
        plt.title('Original Label: {}'.format(label[a].cpu().numpy()),loc ='left')
        plt.imshow(toPil(curr[a].cpu().clone().squeeze()))
        plt.subplot(122)
        plt.title('Adv Label : {}'.format(label_adv[a].cpu().numpy()),loc ='left')
        plt.imshow(toPil(curr_adv[a].cpu().clone().squeeze()))
        plt.show()


def save_data_label(path, test_data_cln, test_data_adv, test_label, test_label_adv):
    if Path(path).exists() == False:
        Path(path).mkdir(parents=True)
    with open(Path(path)/Path('test_data_cln.p'),'wb') as f:
        pickle.dump(test_data_cln.cpu(), f, pickle.HIGHEST_PROTOCOL)

    with open(Path(path)/Path('test_adv(eps_{:.3f}).p'.format(eps)),'wb') as f:
        pickle.dump(test_data_adv.cpu(), f, pickle.HIGHEST_PROTOCOL)

    with open(Path(path)/Path('test_label.p'),'wb') as f:
        pickle.dump(test_label.cpu(), f, pickle.HIGHEST_PROTOCOL)
    
    with open(Path(path)/Path('label_adv(eps_{:.3f}).p'.format(eps)),'wb') as f:
        pickle.dump(test_label_adv.cpu(), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    torch.cuda.set_device(device_id)
    from ResNet import ResNet50
    from VGG import VGG16
    from denseNet import DenseNet
    from Inception_v2 import Inception_v2
    from utils import read_data_label 
    if args.model == 'resnet':
        model = ResNet50(enable_lat =args.enable_lat,
                         epsilon =args.lat_epsilon,
                         pro_num =args.lat_pronum,
                         batch_size =args.model_batchsize,
                         num_classes = 10,
                         if_dropout=args.dropout
                        )
    elif args.model == 'vgg':
        model = VGG16(enable_lat=args.enable_lat,
                      epsilon=args.lat_epsilon,
                      pro_num=args.lat_pronum,
                      batch_size=args.model_batchsize,
                      num_classes=10,
                      if_dropout=args.dropout
                      )
    elif args.model == 'densenet':
        model = DenseNet()
    elif args.model == 'inception':
        model = Inception_v2()
    model.cuda()
    model.load_state_dict(torch.load((args.modelpath)))  
    # if cifar then normalize epsilon from [0,255] to [0,1]
    '''
    if args.dataset == 'cifar10':
        eps = args.attack_epsilon / 255.0
    else:
        eps = args.attack_epsilon
    '''
    eps = args.attack_epsilon
    # the last layer of densenet is F.log_softmax, while CrossEntropyLoss have contained Softmax()
    attack = Attack(dataroot = "/media/dsg3/dsgprivate/lat/data/cifar10/",
                    dataset  = args.dataset,
                    batch_size = args.attack_batchsize,
                    target_model = model,
                    criterion = nn.CrossEntropyLoss(),
                    epsilon = eps,
                    alpha =  args.attack_alpha,
                    iteration = args.attack_iter)
        
    if args.attack == 'fgsm':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.fgsm()
    elif args.attack == 'ifgsm':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.i_fgsm()
    elif args.attack == 'stepll':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.step_ll()
    elif args.attack == 'pgd':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.PGD()
    elif args.attack == 'momentum_ifgsm':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.momentum_ifgsm()
    print(test_data_adv.size(),test_label.size(), type(test_data_adv))
    #test_data, test_label, size = read_data_label('./test_data_cln.p','./test_label.p')
    #test_data_adv, test_label_adv, size = read_data_label('./test_data_cln.p','./test_label.p')
    '''
    test_loader = attack.return_data()
    dataiter = iter(test_loader)
    images,labels = dataiter.next()
    print(images[0])
    '''
    #test_data_cln, test_data_adv, test_label, test_label_adv = attack.i_fgsm()
    #display(test_data_cln, test_data_adv, test_label, test_label_adv)
    if args.generate:
        save_data_label(args.savepath, test_data_cln, test_data_adv,test_label, test_label_adv)
    #save_img(args.imgpath, test_data_cln, test_data_adv, test_label, test_label_adv)



