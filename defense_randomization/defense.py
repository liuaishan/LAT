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
import argparse

device_id = 5

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

# Training settings
parser = argparse.ArgumentParser(description='randomization')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--img_size', type=int, default=32, help='image size of cifar')
parser.add_argument('--img_resize', type=int, default=63, help='image max resize after padding')
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


def main():
    torch.cuda.set_device(device_id)
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
    

    if args.model == 'vgg':
        model = VGG16(enable_lat=False,
                     epsilon=0.8,
                     pro_num=6,
                     batch_size=args.batch_size,
                     if_dropout=True)

    model.cuda()

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print('load model.')
    else:
        print("load failed.")

    model.eval()
    correct = 0
    correct_pad = 0
    total = 0
    for i,(x,y) in enumerate(testing_loader):
        x = x.cuda()
        y = y.cuda()
        # Randomly resize image from 40 to 60
        resize_shape = np.random.randint(40,60)
        resize = transforms.Resize(resize_shape)
        # x: batch x 3 x 32 x 32
        resized_img = torch.Tensor([])
        for i in range(x.size()[0]):
            resized_pil = resize(toPIL(x[i].cpu()))
            resized_img = torch.cat([resized_img,toTensor(resized_pil)])
        resized_img = resized_img.view(args.batch_size,3,resize_shape,resize_shape)
        print(type(resized_img))
        # Randomly padding from rand to 60
        pad_shape = np.random.randint(0, args.img_resize-resize_shape)
        shape = torch.Tensor([args.img_resize,pad_shape,pad_shape])
        padded_img = padding_layer_cifar(resized_img, shape)
        padded_img = padded_img.view(args.batch_size,3,args.img_resize,args.img_resize).cuda()
        print('input size = {}, shape = {}, padded size = {}'.format(resized_img.size(),shape.size(),padded_img.size()))
        h = model(x)
        h_pad = model(padded_img)
        _, pred = torch.max(h.data,1)
        _, pred_pad = torch.max(h_pad.data,1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        correct_pad += (pred_pad == y).sum().item()
    print('Before resizing and padding the Accuracy is : {:.2f} %'.format(100 * correct / total))
    print('After resizing and padding the Accuracy is: {:.2f} %'.format(100 * correct_pad / total))
    model.train()


if __name__ == "__main__":
    main()
