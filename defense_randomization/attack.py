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
import os
import argparse
from VGG import VGG16
from torch.autograd import Function

#device_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

IMG_RESIZE = 35

def defence(inputs):
    x = inputs
    x = x.cuda()
    toPIL = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    # Randomly resize image from 32 to 35
    resize_shape = np.random.randint(32,35)
    resize = transforms.Resize(resize_shape)
    # x: batch x 3 x 32 x 32
    resized_img = torch.Tensor([])
    for i in range(x.size()[0]):
        resized_pil = resize(toPIL(x[i].cpu()))
        resized_img = torch.cat([resized_img,toTensor(resized_pil)])
    resized_img = resized_img.view(args.model_batchsize,3,resize_shape,resize_shape)
    #print(type(resized_img))
    # Randomly padding from rand to 35
    pad_shape = np.random.randint(0, IMG_RESIZE-resize_shape)
    shape = torch.Tensor([IMG_RESIZE,pad_shape,pad_shape])
    
    h_start = shape[1]
    w_start = shape[2]
    output_short = shape[0]
    input_shape = tuple(resized_img.size())
    input_short = torch.min(torch.Tensor(input_shape[2:]))
    input_long = torch.max(torch.Tensor(input_shape[2:]))
    output_long = torch.ceil(output_short * (input_long / input_short))
    output_height = (input_shape[2]>=input_shape[3]) *output_long + (input_shape[2]<input_shape[3]) *output_short
    output_width = (input_shape[2]>=input_shape[3]) *output_short + (input_shape[2]<input_shape[3]) *output_long
    padding = transforms.Pad(padding=((int)(h_start), (int)(w_start), (int)(output_height - h_start - input_shape[2]), 
                                      (int)(output_width - w_start - input_shape[3])),  fill = 0) 
    padded_img = torch.Tensor([])
    for i in range(inputs.size()[0]):
        padded_pil = transforms.ToPILImage()(resized_img[i])
        padding_tmp = transforms.ToTensor()(padding(padded_pil))
        padded_img = torch.cat([padded_img,padding_tmp])     

    padded_img = padded_img.view(args.model_batchsize,3,IMG_RESIZE,IMG_RESIZE)
    #print('input size = {}, shape = {}, padded size = {}'.format(resized_img.size(),shape.size(),padded_img.size()))
    return padded_img

def returnGrad(model,x_adv,y):
    h=model(x_adv)
    loss_func = nn.CrossEntropyLoss()
    loss=loss_func(h,y)
    loss.backward()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    return x_adv.grad
    
    
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
        
    # root of MNIST/CIFAR-10 testset
    def return_data(self):
        if self.dataset == 'mnist':
            test_dataset = torchvision.datasets.MNIST(root=self.dataroot,train=False, transform=transforms.ToTensor())
        elif self.dataset == 'cifar10':
            test_dataset = torchvision.datasets.CIFAR10(root=self.dataroot,train=False, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=False,drop_last=args.droplast)
        return test_loader
        
    def bpda(self):
        test_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0
        for i,(images,labels) in enumerate(test_loader):
            y_true = Variable(labels, requires_grad=False).cuda()
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            fx = defence(x)
            fx = Variable(fx,requires_grad=True).cuda()
            ffx = fx.data
            h = self.model(fx)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()

            for j in range(0, self.iteration):
                h_adv = self.model(fx)
                loss = self.criterion(h_adv, y_true)
                self.model.zero_grad()
                if fx.grad is not None:
                    fx.grad.data.fill_(0)
                loss.backward()
                #grad = returnGrad(model,x_adv,y_true)
                grad = fx.grad
                #I-FGSM
                #x_adv.grad.sign_()   # change the grad with sign ?
                print('step ',i,type(grad),grad.size())
                fx = fx.detach() + self.alpha * torch.sign(grad)
                # according to the paper of Kurakin:
                fx = torch.where(fx > ffx+self.epsilon, ffx+self.epsilon, fx)
                fx = torch.clamp(fx, 0, 1)
                fx = torch.where(fx < ffx-self.epsilon, ffx-self.epsilon, fx)
                fx = torch.clamp(fx, 0, 1)
                fx = Variable(fx.data, requires_grad=True).cuda()

            h_adv = self.model(fx)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()

            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = fx.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, fx.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions == predictions_adv).sum()
            total += len(predictions)
        
        self.model.train()
        error_rate = float(total-correct)*100/total
        print("Error Rate is ",float(total-correct)*100/total)
        print("Before BPDA the accuracy is",float(100*correct_cln)/total)
        print("After BPDA the accuracy is",float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv 


if __name__ == "__main__":

    #torch.cuda.set_device(device_id)

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
        
    test_data_cln, test_data_adv, test_label, test_label_adv = attack.bpda()
    
    print(test_data_adv.size(),test_label.size(), type(test_data_adv))
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



