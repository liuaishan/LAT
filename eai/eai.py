# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
from utils import *

from VGG import VGG16
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable

GPUID = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

model_list = {
#'naive':"/media/dsg3/dsgprivate/yuhang/model/vgg16/naive/naive_param.pkl",
'oat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/oat/naive_param.pkl",
'nat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/nat/naive_param.pkl",
'eat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/eat/naive_param.pkl",
'lat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/aaai/naive_param.pkl",
'dplat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/dplat/lat_param.pkl",
 }
data_list = {
'clean':"/media/dsg3/dsgprivate/lat/test/new/test_data_cln.p",
'fgsm-e8':"/media/dsg3/dsgprivate/lat/test/new/test_adv(eps_0.031).p",
'fgsm-e16':"/media/dsg3/dsgprivate/lat/test/new/test_adv(eps_0.063).p",
'stepll-e8':"/media/dsg3/dsgprivate/lat/test_stepll/test_adv(eps_0.031).p",
'stepll-e16':"/media/dsg3/dsgprivate/lat/test_stepll/test_adv(eps_0.063).p",
'pgd-a16':"/media/dsg3/dsgprivate/lat/test_pgd/test_adv(eps_0.031).p",
'pgd-a2':"/media/dsg3/dsgprivate/lat/test_pgd/test_adv(eps_0.03).p",
'momentum-e8':"/media/dsg3/dsgprivate/lat/test_momentum/vgg/test_adv(eps_0.031).p",
}

lip_list = {
#'naive':list(), # [clean,fgsme8,fgsme16,steplle8,steplle16,pgda16,pgda2,momentume8]
'oat':list(),   # [  0  ,   1  ,   2   ,   3    ,   4     ,   5  ,  6  ,    7     ]
'nat':list(),
'eat':list(),
'lat':list(),
'dplat':list()
}

ENABLE_LAT = False
EPS = 0.3
PROG = 5
BATCH_SIZE = 128
IF_DROP = True

START = 1000
LEN = 100
PATH = '/media/dsg3/dsgprivate/lat/liplist/'

def cal_lip(model,data):
    labelpath = "/media/dsg3/dsgprivate/lat/test/new/test_label.p"
    cnn = VGG16(enable_lat=ENABLE_LAT,
                epsilon=EPS,
                pro_num=PROG,
                batch_size=BATCH_SIZE,
                if_dropout=IF_DROP
                )
    cnn.cuda()
    model_path = model_list[model]
    if os.path.exists(model_path):
        cnn.load_state_dict(torch.load(model_path))
        #print('load model successfully.')
    else:
        print("load failed.")
    model = cnn
    # get test_data , test_label from .p file
    clean_data, test_label, size = read_data_label(data_list['clean'],labelpath)
    test_data, test_label, size = read_data_label(data_list[data],labelpath)
    if size == 0:
        print("reading data failed.")
        return
    if data == 'clean':
        sel_clean = clean_data[START:START+LEN]
        sel_test = test_data[START+LEN:START+2*LEN]
        sel_clean_label = test_label[START:START+LEN]
        sel_test_label = test_label[START+LEN:START+2*LEN]
    else:
        sel_clean = torch.Tensor(LEN,3,32,32)
        sel_test = torch.Tensor(LEN,3,32,32)
        sel_clean_label = torch.LongTensor(LEN)
        sel_test_label = torch.LongTensor(LEN)
        
        j=0
        for i in range(START,test_label.size(0)):
            if test_label[i] == 8:
                sel_clean[j] = clean_data[i]
                sel_test[j] = test_data[i]
                sel_clean_label[j] = 8
                sel_test_label[j] = 8
                j += 1
            if j == LEN:
                break
    '''
    else:
        sel_clean = clean_data[START:START+LEN]
        sel_test = test_data[START:START+LEN]
        sel_clean_label = test_label[START:START+LEN]
        sel_test_label = test_label[START:START+LEN]
    '''

    

    
    # create dataset
    clean_set = Data.TensorDataset(sel_clean, sel_clean_label)
    test_set = Data.TensorDataset(sel_test, sel_test_label)
    
    clean_loader = Data.DataLoader(
        dataset=clean_set,
        batch_size=LEN,  # 1000
        shuffle=False
    )
    
    test_loader = Data.DataLoader(
        dataset=test_set,
        batch_size=LEN,  # 1000
        shuffle=False
    )
    c_lip = 0
    criterion = nn.CrossEntropyLoss()
    # Test the model
    model.eval()
    x_cln = 0
    loss_cln = 0
    for x, y in clean_loader:
        x = x.cuda()
        x_cln = x.view(sel_clean.size(0),-1)
        y = y.cuda()
        #print(y)
        with torch.no_grad():
            h = model(x)
        loss = criterion(h, y)
        loss_cln = loss.item()
    model.train()
    model.eval()
    x_tst = 0
    loss_tst = 0
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        x_tst = x.view(sel_test.size(0),-1)
        with torch.no_grad():
            h = model(x)
        loss = criterion(h, y)
        loss_tst = loss.item()
    model.train()

    dist = 0
    for j in range(sel_test.size(0)):
        dist += torch.max(abs(x_cln[j] - x_tst[j]))     # norm p = inf
    dist = dist / LEN
    c_lip =  abs(loss_cln - loss_tst) / dist 
    
    return c_lip

def generate():
    for model in model_list:
        print('--------- now model is {}:-------------'.format(model))
        for data in data_list:
            c_lip = cal_lip(model,data)
            print('now data is clean <-> {}, lip const is {:.2f}'.format(data,c_lip))
            lip_list[model].append(c_lip)

def save():
    if os.path.exists(PATH) == False:
        os.mkdir(PATH)
    with open(PATH+'lip_list_1.p','wb') as f:
        pickle.dump(lip_list, f, pickle.HIGHEST_PROTOCOL)

def load():
    if os.path.exists(PATH) == False:
        print("load data error")
    with open(PATH+'lip_list_1.p', 'rb') as fr:
        lip_list = pickle.load(fr)
    return lip_list

def concat():
    list1 = dict()
    list2 = dict()
    with open(PATH+'concat/all/'+'ours_all.p', 'rb') as fr:
        lip_list = pickle.load(fr)
    with open(PATH+'concat/all/'+'12ours.p', 'rb') as fr:
        list1 = pickle.load(fr)
    with open(PATH+'concat/all/'+'1oat2nat.p', 'rb') as fr:
        list2 = pickle.load(fr)
    lip_list['dplat'][1] = list1['dplat'][1]
    lip_list['dplat'][2] = list1['dplat'][2]
    lip_list['oat'][1] = list2['oat'][1]
    lip_list['nat'][2] = list2['nat'][2]
    lip_list['nat'][0] = list1['nat'][0]        
    return lip_list

def draw():
    plt.figure(1,(6,4.5))
    x = [0,1,2,3,4,5,6,7]
    plt.plot(x,lip_list['oat'],color='green',label='OAT',marker = 's',ms = 4)
    plt.plot(x,lip_list['nat'],color='orange',label='NAT',marker = 'v',ms = 4.5)
    plt.plot(x,lip_list['eat'],color='royalblue',label='EAT',marker = 'x',ms = 4.5)
    plt.plot(x,lip_list['lat'],color='orchid',label='LAT',marker = '*',ms = 5)
    plt.plot(x,lip_list['dplat'],color='red',label='DP-LAT',marker = 'o',ms = 4.5)
    my_x_ticks = ['clean','FGSM\n('+r'$\epsilon$'+'=8)','FGSM\n('+r'$\epsilon$'+'=16)','Step-LL\n('+r'$\epsilon$'+'=8)','Step-LL\n('+r'$\epsilon$'+'=16)','PGD\n('+r'$\alpha$'+'=16)','PGD\n('+r'$\alpha$'+'=2)','MI-FGSM\n('+r'$\epsilon$'+'=8)']
    plt.xticks(x,my_x_ticks)
    #plt.xlabel('generated examples')
    plt.ylabel(r'$\epsilon$' + '- EAI')
    #plt.title('Robustness evaluation with '+r'$\epsilon$'+' - Empirical Adversarial Insensitivity')
    plt.legend()
    plt.grid(linestyle='--')
    
    plt.savefig("./img/all.pdf")
    plt.show()
    '''
    # scale up lip const
    for model in lip_list:
        tmp_list = lip_list[model]
        for i, clip in enumerate(tmp_list):
            tmp_list[i] = clip * 1e7
    '''
    '''
    # [clean,fgsme8,fgsme16,steplle8,steplle16,pgda16,pgda2,momentume8]
    # [  0  ,   1  ,   2   ,   3    ,   4     ,   5  ,  6  ,    7     ]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9),sharey=True)
    fig.suptitle('Model robustness evaluation with '+r'$\epsilon$'+' - Empirical Adversarial Insensitivity')
    # fgsm
    x00 = [0,1]
    axs[0,0].plot(x00,[lip_list['oat'][1],lip_list['oat'][2]],color='skyblue',label='OAT',marker = 's')
    axs[0,0].plot(x00,[lip_list['nat'][1],lip_list['nat'][2]],color='violet',label='NAT',marker = 'v')
    axs[0,0].plot(x00,[lip_list['eat'][1],lip_list['eat'][2]],color='springgreen',label='EAT',marker = 'x')
    axs[0,0].plot(x00,[lip_list['lat'][1],lip_list['lat'][2]],color='peru',label='LAT',marker = '*')
    axs[0,0].plot(x00,[lip_list['dplat'][1],lip_list['dplat'][2]],color='red',label='DPLAT',marker = 'o')
    my_x00_ticks = [r'$\epsilon$'+'=8',r'$\epsilon$'+'=16']
    axs[0,0].set_xticks(x00)
    axs[0,0].set_xticklabels(my_x00_ticks)
    axs[0,0].set_ylabel(r'$\epsilon$' + '- EAI *({})'.format(SCALE))
    axs[0,0].set_title('FGSM')
    axs[0,0].legend()
    axs[0,0].grid()
    # stepll
    x01 = [0,1,2]
    axs[0,1].plot(x01,[lip_list['oat'][0],lip_list['oat'][3],lip_list['oat'][4]],color='skyblue',label='OAT',marker = 's')
    axs[0,1].plot(x01,[lip_list['nat'][0],lip_list['nat'][3],lip_list['nat'][4]],color='violet',label='NAT',marker = 'v')
    axs[0,1].plot(x01,[lip_list['eat'][0],lip_list['eat'][3],lip_list['eat'][4]],color='springgreen',label='EAT',marker = 'x')
    axs[0,1].plot(x01,[lip_list['lat'][0],lip_list['lat'][3],lip_list['lat'][4]],color='peru',label='LAT',marker = '*')
    axs[0,1].plot(x01,[lip_list['dplat'][0],lip_list['dplat'][3],lip_list['dplat'][4]],color='red',label='DPLAT',marker = 'o')
    my_x01_ticks = ['clean',r'$\epsilon$'+'=8',r'$\epsilon$'+'=16']
    axs[0,1].set_xticks(x01)
    axs[0,1].set_xticklabels(my_x01_ticks)
    axs[0,1].set_title('Step-LL')
    axs[0,1].legend()
    axs[0,1].grid()
    # pgd
    x10 = [0,1,2]
    axs[1,0].plot(x10,[lip_list['oat'][0],lip_list['oat'][5],lip_list['oat'][6]],color='skyblue',label='OAT',marker = 's')
    axs[1,0].plot(x10,[lip_list['nat'][0],lip_list['nat'][5],lip_list['nat'][6]],color='violet',label='NAT',marker = 'v')
    axs[1,0].plot(x10,[lip_list['eat'][0],lip_list['eat'][5],lip_list['eat'][6]],color='springgreen',label='EAT',marker = 'x')
    axs[1,0].plot(x10,[lip_list['lat'][0],lip_list['lat'][5],lip_list['lat'][6]],color='peru',label='LAT',marker = '*')
    axs[1,0].plot(x10,[lip_list['dplat'][0],lip_list['dplat'][5],lip_list['dplat'][6]],color='red',label='DPLAT',marker = 'o')
    my_x10_ticks = ['clean',r'$\epsilon$'+'=8\n'+r'$\alpha$'+'=16',r'$\epsilon$'+'=8\n'+r'$\alpha$'+'=2']
    axs[1,0].set_xticks(x10)
    axs[1,0].set_xticklabels(my_x10_ticks)
    #axs[1,0].set_xlabel("data")
    axs[1,0].set_ylabel(r'$\epsilon$' + '- EAI *({})'.format(SCALE))
    axs[1,0].set_title('PGD')
    axs[1,0].legend()
    axs[1,0].grid()
    # momentum
    x11 = [0,1]
    axs[1,1].plot(x11,[lip_list['oat'][0],lip_list['oat'][7]],color='skyblue',label='OAT',marker = 's')
    axs[1,1].plot(x11,[lip_list['nat'][0],lip_list['nat'][7]],color='violet',label='NAT',marker = 'v')
    axs[1,1].plot(x11,[lip_list['eat'][0],lip_list['eat'][7]],color='springgreen',label='EAT',marker = 'x')
    axs[1,1].plot(x11,[lip_list['lat'][0],lip_list['lat'][7]],color='peru',label='LAT',marker = '*')
    axs[1,1].plot(x11,[lip_list['dplat'][0],lip_list['dplat'][7]],color='red',label='DPLAT',marker = 'o')
    my_x11_ticks = ['clean',r'$\epsilon$'+'=8\n'+r'$\alpha$'+'=1']
    axs[1,1].set_xticks(x11)
    axs[1,1].set_xticklabels(my_x11_ticks)
    #axs[1,1].set_xlabel("data")
    axs[1,1].set_title('MI-FGSM')
    axs[1,1].legend()
    axs[1,1].grid()
    
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.90, hspace=0.25,wspace=0.20)
    
    fig.savefig('./img/img4.png')
    plt.show()
    '''
def show_clean():
    for model in lip_list:
        data = lip_list[model][0]
        print('model is {}, clean-clean is {}'.format(model,data))

if __name__ == "__main__":
    #generate()
    #save()
    #lip_list = load()
    lip_list = concat()
    draw()
    show_clean()

