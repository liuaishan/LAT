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
'oat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/oat/naive_param.pkl",
'nat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/nat/naive_param.pkl",
'eat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/eat/naive_param.pkl",
'lat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/aaai/naive_param.pkl",
'dplat':"/media/dsg3/dsgprivate/yuhang/model/vgg16/dplat/lat_param.pkl",
}

data_list = {
'fgsm':"/media/dsg3/dsgprivate/lat/test_lip/fgsm/",
'stepll':"/media/dsg3/dsgprivate/lat/test_lip/stepll/",
}

clean_list = {    
'oat':0,   
'nat':0,   # eps = [4,5,6,7,8,9,10,11,12,13,14,15,16]
'eat':0,
'lat':0,
'dplat':0
}

lip_list = {    
'oat':list(),   
'nat':list(),   # eps = [4,5,6,7,8,9,10,11,12,13,14,15,16]
'eat':list(),
'lat':list(),
'dplat':list()
}

TYPE = 'stepll'

START = 8200
LEN = 1000

ENABLE_LAT = False
EPS = 0.3
PROG = 5
BATCH_SIZE = 128
IF_DROP = True

PATH = '/media/dsg3/dsgprivate/lat/liplist/'

def cal_lip(model_path,data_path):
    cleanpath = "/media/dsg3/dsgprivate/lat/test_lip/"+TYPE+"/test_data_cln.p"
    labelpath = "/media/dsg3/dsgprivate/lat/test_lip/"+TYPE+"/test_label.p"
    model = VGG16(enable_lat=ENABLE_LAT,
                epsilon=EPS,
                pro_num=PROG,
                batch_size=BATCH_SIZE,
                if_dropout=IF_DROP
                )
    model.cuda()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        #print('load model successfully.')
    else:  
        print("load failed.")
    
    # get test_data , test_label from .p file
    clean_data, test_label, size = read_data_label(cleanpath,labelpath)
    test_data, test_label, size = read_data_label(data_path,labelpath)

    if size == 0:
        print("reading data failed.")
        return
    
    sel_clean = torch.Tensor(LEN,3,32,32)
    sel_test = torch.Tensor(LEN,3,32,32)
    sel_clean_label = torch.LongTensor(LEN)
    sel_test_label = torch.LongTensor(LEN)
    j=0
    for i in range(START,test_label.size(0)):
        if test_label[i] == 3:
            sel_clean[j] = clean_data[i]
            sel_test[j] = test_data[i]
            sel_clean_label[j] = 3
            sel_test_label[j] = 3
            j += 1
        if j == LEN:
            break
    '''
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
        batch_size=sel_clean.size(0),  # LEN
        shuffle=False
    )
    
    test_loader = Data.DataLoader(
        dataset=test_set,
        batch_size=sel_test.size(0),  # LEN
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
        model_path = model_list[model]
        path = data_list[TYPE]
        for eps in range(4,16+1):
            data_path = path + 'test_adv(eps_{}).p'.format(eps)
            c_lip = cal_lip(model_path,data_path)
            print('now data is {}-eps-{}, lip const is {:.2f}'.format(TYPE,eps,c_lip))
            lip_list[model].append(c_lip)

def save():
    if os.path.exists(PATH) == False:
        os.mkdir(PATH)
    with open(PATH+TYPE+'/clean_list.p','wb') as f:
        pickle.dump(clean_list, f, pickle.HIGHEST_PROTOCOL)
    with open(PATH+TYPE+'/lip_list_2.p','wb') as f:
        pickle.dump(lip_list, f, pickle.HIGHEST_PROTOCOL)
        
def load():
    if os.path.exists(PATH) == False:
        print("load data error")
    with open(PATH+TYPE+'/clean_list.p', 'rb') as fr:
        clean_list = pickle.load(fr)
    with open(PATH+TYPE+'/lip_list_2.p', 'rb') as fr:
        lip_list = pickle.load(fr)
    return clean_list,lip_list

def concat():
    list1 = dict()
    list2 = dict()
    if TYPE == 'fgsm':
        with open(PATH+'concat/'+TYPE+'/ours.p', 'rb') as fr:
            lip_list = pickle.load(fr)
        with open(PATH+'concat/'+TYPE+'/nat.p', 'rb') as fr:
            list1 = pickle.load(fr)
        #with open(PATH+'concat/'+TYPE+'/456eat.p', 'rb') as fr:
        #    list2 = pickle.load(fr)
        lip_list['nat'] = list1['nat'] 
        #print(lip_list['dplat'])   
        return lip_list
    elif TYPE == 'stepll':
        with open(PATH+'concat/'+TYPE+'/ours.p', 'rb') as fr:
            lip_list = pickle.load(fr)
        with open(PATH+'concat/'+TYPE+'/nat.p', 'rb') as fr:
            list1 = pickle.load(fr)
        lip_list['nat'] = list1['nat']
        return lip_list
def draw():
    plt.figure(1,(6,4.5))
    x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    plt.plot(x,lip_list['oat'],color='green',label='OAT',marker = 's',ms = 3)#linestyle='-.')#,##
    plt.plot(x,lip_list['nat'],color='orange',label='NAT',marker = 'v',ms = 4)#linestyle='-.')###,
    plt.plot(x,lip_list['eat'],color='royalblue',label='EAT',marker = 'x',ms = 4)#linestyle='-.')###,
    plt.plot(x,lip_list['lat'],color='orchid',label='LAT',marker = '*',ms = 4)#linestyle='-.')###,
    plt.plot(x,lip_list['dplat'],color='red',label='DP-LAT',marker = 'o',ms = 3)#linestyle='-.')###,
    my_x_ticks = np.arange(4,16+1,1)
    plt.xticks(x,my_x_ticks)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\epsilon$' + '- EAI')
    #plt.title(TYPE)
    plt.legend()
    plt.grid(linestyle='--')
    #plt.title('Model robustness evaluation with '+r'$\epsilon$'+' - Empirical Adversarial Insensitivity')
    plt.savefig("./img/"+TYPE+".pdf")
    plt.show()

def clean(model_path):
    cleanpath = "/media/dsg3/dsgprivate/lat/test_lip/"+TYPE+"/test_data_cln.p"
    labelpath = "/media/dsg3/dsgprivate/lat/test_lip/"+TYPE+"/test_label.p"
    
    model = VGG16(enable_lat=ENABLE_LAT,
                epsilon=EPS,
                pro_num=PROG,
                batch_size=BATCH_SIZE,
                if_dropout=IF_DROP
                )
    model.cuda()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        #print('load model successfully.')
    else:  
        print("load failed.")
    
    # get test_data , test_label from .p file
    clean_data, test_label, size = read_data_label(cleanpath,labelpath)
    
    if size == 0:
        print("reading data failed.")
        return
    
    sel_clean = clean_data[START:START+LEN]
    sel_test = clean_data[START+LEN:START+2*LEN]
    sel_clean_label = test_label[START:START+LEN]
    sel_test_label = test_label[START+LEN:START+2*LEN]
    
    # create dataset
    clean_set = Data.TensorDataset(sel_clean, sel_clean_label)
    test_set = Data.TensorDataset(sel_test, sel_test_label)
    clean_loader = Data.DataLoader(
        dataset=clean_set,
        batch_size=sel_clean.size(0),  # LEN
        shuffle=False
    )
    test_loader = Data.DataLoader(
        dataset=test_set,
        batch_size=sel_test.size(0),  # LEN
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
    
def gen_clean():
    for model in model_list:
        model_path = model_list[model]
        clean_list[model] = clean(model_path)

def show_clean():
    for model in clean_list:
        data = clean_list[model]
        print('model is {}, clean-clean is {}'.format(model,data))

if __name__ == "__main__":
    #generate()
    #gen_clean()
    #save()
    #clean_list, lip_list = load()
    lip_list = concat()
    #show_clean()
    draw()

