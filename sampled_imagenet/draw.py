import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('bmh')
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

def read_data_label(data_path, label_path):

    if not os.path.exists(data_path):
        return None, None, 0

    with open(data_path, 'rb') as fr:
        test_data = pickle.load(fr)
        size = len(test_data)
    with open(label_path, 'rb') as fr:
        test_label = pickle.load(fr)
    return test_data, test_label, size
    
def save_img(imgpath,test_data_cln, test_data_adv, test_label, test_label_adv):
    #save adversarial example
    if os.path.exists(imgpath) == False:
        os.mkdir(imgpath)
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
        im.save(imgpath+'{}_label_{}_cln.jpg'.format(i,test_label[i]))
        im = toImg(image_adv[i])
        #im.show()
        im.save(imgpath+'{}_label_{}_adv.jpg'.format(i,test_label_adv[i]))

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
        
if __name__ == "__main__":
    cln_path = "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/" + "test_data_cln.p"
    adv_path = "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/" + "test_adv(eps_0.031).p"
    cln_label_path = "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/" + "test_label.p"
    adv_label_path = "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/" + "label_adv(eps_0.031).p"
    # get test_data , test_label from .p file
    cln_data, cln_label, size = read_data_label(cln_path,cln_label_path)
    adv_data, adv_label, size = read_data_label(adv_path,adv_label_path)
    if size == 0:
        print("reading data failed.")
        exit()
    print("data successfully loaded.")
    img_path = "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/resnet18/" + "img/"
    save_img(img_path,cln_data,adv_data,cln_label,adv_label)
    display(cln_data,adv_data,cln_label,adv_label)