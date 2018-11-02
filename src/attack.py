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


def fgsm(model, criterion, eps=0.3):
    model.eval()
    
    images_all = list()
    adv_all = list()
    correct = 0
    correct_cln = 0
    correct_adv = 0
    total = 0 
    for images, labels in test_loader:
        x = Variable(images, requires_grad = True)
        y_true = Variable(labels, requires_grad = False)

        h = model(x)
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
        
        h_adv = model(x_adv)
        _, predictions_adv = torch.max(h_adv,1)
        correct_adv += (predictions_adv == labels).sum()

        images_all.append([x.data.view(-1,28,28).detach().cpu(), labels])
        adv_all.append([x_adv.data.view(-1,28,28).cpu(), predictions_adv])

        correct += (predictions == predictions_adv).sum()
        total += len(predictions)
    
    model.train()
    
    error_rate = float(total-correct)*100/total
    print("Error Rate is ", float(total-correct)*100/total)
    print("Before FGSM the accuracy is", float(100*correct_cln)/total)
    print("After FGSM the accuracy is", float(100*correct_adv)/total)

    return images_all, adv_all, error_rate

def save(images_all, adv_all):
    #save adversarial examples
    image, label = images_all[0]
    image_adv, label_adv = adv_all[0]
    tot = len(image)
    for i in range(0, tot):
        im = toImg(image[i].unsqueeze(0))
        im.save(Path('img/eps_{}/{}_clean.jpg'.format(eps,i)))
        im = toImg(image_adv[i].unsqueeze(0))
        im.save(Path('img/eps_{}/{}_adver.jpg'.format(eps,i)))

def display(images_all, adv_all):
    # display a batch adv
    curr, label = images_all[0]
    curr_adv, label_adv = adv_all[0]
    disp_batch = 10
    for a in range(disp_batch):
        plt.figure()
        plt.subplot(121)
        plt.title('Original Label: {}'.format(label[a].cpu().numpy()),loc ='left')
        plt.imshow(curr[a].numpy(),cmap='gray')
        plt.subplot(122)
        plt.title('Adv Label : {}'.format(label_adv[a].cpu().numpy()),loc ='left')
        plt.imshow(curr_adv[a].numpy(),cmap='gray')
        plt.show()
    total = batch_size
    correct = (label==label_adv).sum()
    print("Batch Error rate ",float(total-correct)*100/total)

if __name__ == "__main__":
    eps = 0.3
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''
    model = LeNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    '''
    #train()
    model.load_state_dict(torch.load('model.pth'))
    #test()
    images_all, adv_all, error_rate = fgsm(model,criterion,eps)



