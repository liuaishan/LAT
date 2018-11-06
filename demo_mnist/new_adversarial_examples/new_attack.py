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
from LeNet import LeNet
import argparse
import pickle
import torch.utils.data as Data
import math

parser = argparse.ArgumentParser(description='lat implementation')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--train_data_path', default=".\\data\\", help='training dataset path')
parser.add_argument('--model_path', default=".\\model\\", help='number of classes')
parser.add_argument('--batchnorm', default=True, help='batch normalization')
parser.add_argument('--dropout', default=True, help='dropout')
parser.add_argument('--dataset', default='mnist', help='data set')
parser.add_argument('--model', default='lenet', help='target model')

args = parser.parse_args()

print(args)

#generate adversarial examples
def fgsm(model, criterion, eps=0.3):
    model.eval()
    
    images_all = list()
    adv_all = list()
    correct_cln = 0
    correct_adv = 0
    total = 0 
    train_data = torchvision.datasets.MNIST(  #training_set
                 root=args.train_data_path,
                 train=True,
                 transform=torchvision.transforms.ToTensor(),
                 download=True
                 )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=False)
    for images, labels in train_loader: 
        x = Variable(images, requires_grad = True).cuda()
        y_true = Variable(labels, requires_grad = False).cuda()

        h,_ = model(x)
        _, predictions = torch.max(h,1)
        correct_cln += (predictions == y_true).sum()
        loss = criterion(h, y_true)
        model.zero_grad()
        if x.grad is not None:
            x.grad.data.fill_(0)
        loss.backward()
        
        #FGSM
        #x.grad.sign_()   # change the grad with sign ?
        x_adv = x.detach() + eps * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv,0,1)
        
        h_adv,_ = model(x_adv)
        _, predictions_adv = torch.max(h_adv,1)
        correct_adv += (predictions_adv == y_true).sum()

		#this part should store the X_original + Y_true
        images_all.append([x.data.view(-1,28,28).detach().cpu(), labels])
		#this part should store the X_adv + Y_true
        adv_all.append([x_adv.data.view(-1,28,28).cpu(), labels])

        total += len(predictions)
    
    model.train()
    
    print("Before FGSM the accuracy is", float(100*correct_cln)/total)
    print("After FGSM the accuracy is", float(100*correct_adv)/total)
    print(len(adv_all))
    return images_all, adv_all

#save adversarial and original images
def save(images_all, adv_all,eps):
    toImg = transforms.ToPILImage()
    #save adversarial examples
    lenth = len(images_all)
    for i in range(lenth):
        image, label = images_all[i]
        image_adv, label_adv = adv_all[i]
        tot = len(image)
        for j in range(0, tot):
            im = toImg(image[j].unsqueeze(0))
            im.save(Path('.\\image\\clean\\eps_{}\\{}.jpg'.format(eps,j)))
            im = toImg(image_adv[j].unsqueeze(0))
            im.save(Path('.\\image\\adver\\eps_{}\\{}.jpg'.format(eps,j)))



#save adversarial example into .p file
def save_data(images_all, adv_all,eps):
    lenth = len(adv_all)
    X = adv_all[0][0]
    Y = adv_all[0][1]
    for i in range(1,lenth):
        X_BATCH = adv_all[i][0]
        Y_BATCH = adv_all[i][1]
        X = torch.cat((X,X_BATCH),0)
        Y = torch.cat((Y,Y_BATCH),0)
    print(X.size())
    print(Y.size())
    with open('eps_{}.p'.format(eps),'wb') as f:
        pickle.dump([X,Y], f, pickle.HIGHEST_PROTOCOL)	

#save adversarial example mixed with original data into .p file
#alpha : the proportion of adversarial examples
def save_mixed_example(alpha,images_all,adv_all,eps,batch_size,total_num):
    adv_size = math.floor(batch_size * alpha)
    adv_x = adv_all[0][0][0:adv_size]
    adv_y = adv_all[0][1][0:adv_size]
    orig_x = images_all[0][0][adv_size:]
    orig_y = images_all[0][1][adv_size:]	
    X = torch.cat((adv_x,orig_x),0)
    Y = torch.cat((adv_y,orig_y),0) 
	
    batch_num = math.floor(total_num/batch_size) + 1
    for i in range(1,batch_num):
        if(i != (batch_num-1)):
            adv_size = math.floor(batch_size * alpha)
        else:
            adv_size = math.floor((total_num-(batch_num-1)*batch_size) * alpha)
        adv_x = adv_all[i][0][0:adv_size]
        adv_y = adv_all[i][1][0:adv_size]
        orig_x = images_all[i][0][adv_size:]
        orig_y = images_all[i][1][adv_size:]	
        X = torch.cat((X,adv_x,orig_x),0)
        Y = torch.cat((Y,adv_y,orig_y),0) 
    print(X.size())
    print(Y.size())
    with open('mixed_eps_{}.p'.format(eps),'wb') as f:
        pickle.dump([X,Y], f, pickle.HIGHEST_PROTOCOL)    	


if __name__ == "__main__":
    eps = 0.3
    
    model = LeNet(enable_lat= False,
                    epsilon= 0,
                    pro_num= 1,
                    batch_size=args.batchsize,
                    batch_norm=args.batchnorm,
                    if_dropout=args.dropout).cuda()
    criterion = nn.CrossEntropyLoss()
	
    real_model_path = args.model_path + "naive_param.pkl"
	
    #train()
    model.load_state_dict(torch.load(real_model_path))
    #test()
    images_all, adv_all= fgsm(model,criterion,eps)
	
    #save_data(images_all, adv_all,eps)
    save_mixed_example(0.5,images_all, adv_all,eps,64,60000)

	



