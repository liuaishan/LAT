#---------------------------------------------------------------------------------------
# Mitigating Adversarial Effects Through Randomization (on CIFAR-10), edited by Hang
# NIPS 2017 defense rank No.2
# use VGG model for implementation
#---------------------------------------------------------------------------------------
import os

import random

import numpy as np
from utils import read_data_label
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from VGG import VGG16
from ResNet import ResNet50
import argparse

GPUID = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

# Training settings
parser = argparse.ArgumentParser(description='randomization')
parser.add_argument('--test_flag', type=get_bool, default=False, help='test all adv or defense only')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--img_size', type=int, default=32, help='image size of cifar')
parser.add_argument('--img_resize', type=int, default=63, help='image max resize after padding')
parser.add_argument('--iter_times', type=int, default=10, help='iter times with resizing and padding')
parser.add_argument('--num_classes', type=int, default=10, help='number of classified classes')
parser.add_argument('--test_data_path', default="/media/dsg3/dsgprivate/lat/test/new/test_adv(eps_0.031).p", help='test images path')
parser.add_argument('--test_label_path', default="/media/dsg3/dsgprivate/lat/test/new/test_label.p", help='test labels path')
parser.add_argument('--model', default='vgg', help='test model:vgg, resnet,...')
parser.add_argument('--model_path', default="/media/dsg3/dsgprivate/lat/liuaishan/cifar10/vgg16_origin_dropout/naive_param.pkl", help='test model path')
args = parser.parse_args()
#print(args)

# img_resize = max padding resize (60)
# resize_shape = random resize val (40 ~ 60)
# inputs = batch x 3 x resize_shape x resize_shape 
# shape = img_resize x (0 ~ img_resize-resize_shape) x (0 ~ img_resize-resize_shape) 
# transforms.Pad((left,up,right,down),0)
def padding_layer_cifar(inputs, shape):
    if not args.test_flag:
        print(inputs.size(),shape)
    h_start = shape[1]
    w_start = shape[2]
    output_short = shape[0]
    input_shape = tuple(inputs.size())
    input_short = torch.min(torch.Tensor(input_shape[2:]))
    input_long = torch.max(torch.Tensor(input_shape[2:]))
    output_long = torch.ceil(output_short * (input_long / input_short))
    output_height = (input_shape[2]>=input_shape[3]) *output_long + (input_shape[2]<input_shape[3]) *output_short
    output_width = (input_shape[2]>=input_shape[3]) *output_short + (input_shape[2]<input_shape[3]) *output_long
    #print('h_start={},w_start={},output_short={},input_short={},input_long={},output_long={}'.format(h_start,w_start,output_short,input_short,input_long,output_long))
    #print('output height={}, input_shape[2]={}, output width={}, input_shape[3]={}'.format(output_height,input_shape[2],output_width,input_shape[3]))
    #print((int)(h_start), (int)(w_start), (int)(output_height-h_start-input_shape[2]),(int)(output_width-w_start-input_shape[3]))
    padding = transforms.Pad(padding=((int)(h_start), (int)(w_start), (int)(output_height - h_start - input_shape[2]), (int)(output_width - w_start - input_shape[3])),fill=0)
    padded_img = torch.Tensor([])
    for i in range(inputs.size()[0]):
        padded_pil = transforms.ToPILImage()(inputs[i])
        #print(inputs[i].size())
        padding_tmp = transforms.ToTensor()(padding(padded_pil))
        #print(padding_tmp.size())
        padded_img = torch.cat([padded_img,padding_tmp]) 
    return padded_img

def test_op(model):
    toPIL = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    #Read .p files from input directory in batches.
    #batch_shape =  batch x 3 x 32 x 32
    test_data, test_label, size = read_data_label(args.test_data_path,args.test_label_path)

    if size == 0:
        print("reading data failed.")
        return

    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(dataset=testing_set,batch_size=args.batch_size,shuffle=False,drop_last=True)
    
    model.eval()
    correct = 0
    correct_pad = 0
    total = 0
    for i,(x,y) in enumerate(testing_loader):
        x = x.cuda()
        y = y.cuda()
        h = model(x)
        _, pred = torch.max(h.data,1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        h_pad = torch.Tensor(args.batch_size,args.num_classes,args.iter_times)
        for j in range(args.iter_times):
            # Randomly resize image from 32 to 35
            resize_shape = np.random.randint(args.img_size,args.img_resize)
            resize = transforms.Resize(resize_shape)
            # x: batch x 3 x 32 x 32
            resized_img = torch.Tensor([])
            for k in range(x.size()[0]):
                resized_pil = resize(toPIL(x[k].cpu()))
                resized_img = torch.cat([resized_img,toTensor(resized_pil)])
            resized_img = resized_img.view(args.batch_size,3,resize_shape,resize_shape)
            if not args.test_flag:
                print(type(resized_img))
            # Randomly padding from rand to 35
            pad_shape = np.random.randint(0, args.img_resize-resize_shape)
            shape = torch.Tensor([args.img_resize,pad_shape,pad_shape])
            padded_img = padding_layer_cifar(resized_img, shape)
            padded_img = padded_img.view(args.batch_size,3,args.img_resize,args.img_resize).cuda()
            if not args.test_flag:
                print('input size = {}, shape = {}, padded size = {}'.format(resized_img.size(),shape.size(),padded_img.size()))
            h_pad_iter = model(padded_img)
            h_pad[:,:,j] = h_pad_iter
        h_pad = torch.mean(h_pad,dim=-1).cuda()
        _, pred_pad = torch.max(h_pad.data,1)
        correct_pad += (pred_pad == y).sum().item()
    print('Before resizing and padding the Accuracy is : {:.2f} %'.format(100 * correct / total))
    print('After resizing and padding the Accuracy is: {:.2f} %'.format(100 * correct_pad / total))
    model.train()
    
def test_one(model,data_cat,data_path):
    toPIL = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
        
    label_path = "/media/dsg3/dsgprivate/lat/test/resnet/test_label.p"
    test_data, test_label, size = read_data_label(data_path,label_path)

    if size == 0:
        print("reading data failed.")
        return
    
    test_data = test_data.cuda()
    test_label = test_label.cuda()
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    
    model.eval()
    correct = 0
    correct_pad = 0
    total = 0
    for i,(x,y) in enumerate(testing_loader):
        x = x.cuda()
        y = y.cuda()
        h = model(x)
        _, pred = torch.max(h.data,1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        h_pad = torch.Tensor(args.batch_size,args.num_classes,args.iter_times)
        for j in range(args.iter_times):
            # Randomly resize image from 32 to 35
            resize_shape = np.random.randint(args.img_size,args.img_resize)
            resize = transforms.Resize(resize_shape)
            # x: batch x 3 x 32 x 32
            resized_img = torch.Tensor([])
            for k in range(x.size()[0]):
                resized_pil = resize(toPIL(x[k].cpu()))
                resized_img = torch.cat([resized_img,toTensor(resized_pil)])
            resized_img = resized_img.view(args.batch_size,3,resize_shape,resize_shape)
            if not args.test_flag:
                print(type(resized_img))
            # Randomly padding from rand to 35
            pad_shape = np.random.randint(0, args.img_resize-resize_shape)
            shape = torch.Tensor([args.img_resize,pad_shape,pad_shape])
            padded_img = padding_layer_cifar(resized_img, shape)
            padded_img = padded_img.view(args.batch_size,3,args.img_resize,args.img_resize).cuda()
            if not args.test_flag:
                print('input size = {}, shape = {}, padded size = {}'.format(resized_img.size(),shape.size(),padded_img.size()))
            h_pad_iter = model(padded_img)
            h_pad[:,:,j] = h_pad_iter
        h_pad = torch.mean(h_pad,dim=-1).cuda()
        _, pred_pad = torch.max(h_pad.data,1)
        correct_pad += (pred_pad == y).sum().item()

    #print('Before Padding Model Acc on {} : {:.2f} %'.format( data_cat,(100 * correct / total)) )  
    print('After Padding Model Acc on {} : {:.2f} %'.format( data_cat,(100 * correct_pad / total)) )        
    model.train(True)

def test_all(model):

    model_list = ['vgg','resnet','densenet','inception']
    adv_data_vgg = {
    'fgsm-e8-vgg':"/media/dsg3/dsgprivate/lat/test/new/test_adv(eps_0.031).p",
    'fgsm-e16-vgg':"/media/dsg3/dsgprivate/lat/test/new/test_adv(eps_0.063).p",
    'stepll-e8-vgg':"/media/dsg3/dsgprivate/lat/test_stepll/test_adv(eps_0.031).p",
    'stepll-e16-vgg':"/media/dsg3/dsgprivate/lat/test_stepll/test_adv(eps_0.063).p",
    'pgd-a16-vgg':"/media/dsg3/dsgprivate/lat/test_pgd/test_adv(eps_0.031).p",
    'pgd-a2-vgg':"/media/dsg3/dsgprivate/lat/test_pgd/test_adv(eps_0.03).p",
    'momentum-e8-vgg':"/media/dsg3/dsgprivate/lat/test_momentum/vgg/test_adv(eps_0.031).p",
    }
    adv_data_resnet = {
    'fgsm-e8-resnet':"/media/dsg3/dsgprivate/lat/test/resnet/test_adv(eps_0.031).p",
    'fgsm-e16-resnet':"/media/dsg3/dsgprivate/lat/test/resnet/test_adv(eps_0.063).p",
    'stepll-e8-resnet':"/media/dsg3/dsgprivate/lat/test_stepll/resnet/test_adv(eps_0.031).p",
    'stepll-e16-resnet':"/media/dsg3/dsgprivate/lat/test_stepll/resnet/test_adv(eps_0.063).p",
    'pgd-a16-resnet':"/media/dsg3/dsgprivate/lat/test_pgd/resnet/test_adv(eps_0.031_a_0.063).p",
    'pgd-a2-resnet':"/media/dsg3/dsgprivate/lat/test_pgd/resnet/test_adv(eps_0.031_a_0.008).p",
    'momentum-e8-resnet':"/media/dsg3/dsgprivate/lat/test_momentum/resnet/test_adv(eps_0.031).p",
    }
    adv_data_densenet = {
    'fgsm-e8-densenet':"/media/dsg3/dsgprivate/lat/test/densenet/test_adv(eps_0.031).p",
    'fgsm-e16-densenet':"/media/dsg3/dsgprivate/lat/test/densenet/test_adv(eps_0.063).p",
    'stepll-e8-densenet':"/media/dsg3/dsgprivate/lat/test_stepll/densenet/test_adv(eps_0.031).p",
    'stepll-e16-densenet':"/media/dsg3/dsgprivate/lat/test_stepll/densenet/test_adv(eps_0.063).p",
    'pgd-a16-densenet':"/media/dsg3/dsgprivate/lat/test_pgd/densenet/test_adv(eps_0.031_a_0.063).p",
    'pgd-a2-densenet':"/media/dsg3/dsgprivate/lat/test_pgd/densenet/test_adv(eps_0.031_a_0.008).p",
    'momentum-e8-densenet':"/media/dsg3/dsgprivate/lat/test_momentum/densenet/test_adv(eps_0.031).p",
    }
    adv_data_inception = {
     'fgsm-e8-inception':"/media/dsg3/dsgprivate/lat/test/inception/test_adv(eps_0.031).p",
    'fgsm-e16-inception':"/media/dsg3/dsgprivate/lat/test/inception/test_adv(eps_0.063).p",
    'stepll-e8-inception':"/media/dsg3/dsgprivate/lat/test_stepll/inception/test_adv(eps_0.031).p",
    'stepll-e16-inception':"/media/dsg3/dsgprivate/lat/test_stepll/inception/test_adv(eps_0.063).p",
    'pgd-a16-inception':"/media/dsg3/dsgprivate/lat/test_pgd/inception/test_adv(eps_0.031_a_0.063).p",
    'pgd-a2-inception':"/media/dsg3/dsgprivate/lat/test_pgd/inception/test_adv(eps_0.031_a_0.008).p",
    'momentum-e8-inception':"/media/dsg3/dsgprivate/lat/test_momentum/inception/test_adv(eps_0.031).p",   
    }
    
    print('-------------------Now is clean data------------------------------')
    test_op(model)
    
    for target in model_list:
        print('------Now adv data come from {} -------'.format(target))
  
        if target == 'vgg':
            data_list = adv_data_vgg
        elif target == 'resnet':
            data_list = adv_data_resnet
        elif target == 'densenet':
            data_list = adv_data_densenet
        elif target == 'inception':
            data_list = adv_data_inception

        for data_cat in data_list:
            data_path = data_list[data_cat]
            test_one(model,data_cat,data_path)

if __name__ == "__main__":

    if args.model == 'vgg':
        model = VGG16(enable_lat=False,
                     epsilon=0.6,
                     pro_num=5,
                     batch_size=args.batch_size,
                     if_dropout=True)
    elif args.model == 'resnet':
        model = ResNet50(enable_lat=False,
                     epsilon=0.6,
                     pro_num=5,
                     batch_size=args.batch_size,
                     if_dropout=True)    
    model.cuda()

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print('load model.')
    else:
        print("load failed.")
    if args.test_flag:
        test_all(model)
    else:
        test_op(model)
