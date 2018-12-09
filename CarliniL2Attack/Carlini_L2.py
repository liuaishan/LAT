#--------------------------------------------------------------------------------------------------------
#PyTorch Carlini and Wagner L2 attack algorithm.
#Based on paper by Carlini & Wagner, https://arxiv.org/abs/1608.04644 and a reference implementation at
#https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
#-------------------------------------------------------------------------------------------------------
# Referenced from https://github.com/alekseynp/nips2017-adversarial/, edited by Hang.
#--------------------------------------------------------------------------------------------------------

import os
import sys
import torch
import numpy as np
from torch import optim
from torch import autograd
from helpers import *
import VGG
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

NUM_CLASSES = 10
CONFIDENCE = 5   # FIXME need to find a good value for this, 0 value used in paper not doing much...
INITIAL_CONST = 0.05 # bumped up from default of .01 in reference code
SEARCH_STEPS = 4
MAX_STEPS = 1000
LEARNING_RATE = 0.01
ABORT_EARLY = True
X_MIN = 0.
X_MAX = 1.
TARGETED = False
DEBUG = False
CUDA = True
#--------------------------------------------------------------------------------------------------------
# Initial_const++: init loss++ convergence speed - Failure none ; 
# confidence++: total loss++, convergence speed --, Failure ++ ;
# search steps++: loss convergence ++; 
#--------------------------------------------------------------------------------------------------------
'''
#settings in original paper :
BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess
'''
MODEL = 'vgg'
DATASET = 'cifar10'
DATA_ROOT = "/media/dsg3/dsgprivate/lat/data/cifar10/"
BATCHSIZE = 128
DROP_LAST = False
MODEL_PATH = "/media/dsg3/dsgprivate/lat/liuaishan/cifar10/vgg16_origin_dropout/naive_param.pkl"

# root of MNIST/CIFAR-10 testset
def return_data():
    if DATASET == 'mnist':
        test_dataset = torchvision.datasets.MNIST(root=DATA_ROOT,train=False, transform=transforms.ToTensor())
    elif DATASET == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT,train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCHSIZE,shuffle=False,drop_last=DROP_LAST)
    return test_loader

class AttackCarliniWagnerL2:

    def __init__(self, targeted=False, search_steps=None, max_steps=None, cuda=True, debug=False):
        self.debug = DEBUG
        self.targeted = TARGETED
        self.num_classes = NUM_CLASSES
        self.confidence = CONFIDENCE  
        self.initial_const = INITIAL_CONST  
        self.binary_search_steps = SEARCH_STEPS
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = MAX_STEPS
        self.abort_early = ABORT_EARLY
        self.clip_min = X_MIN
        self.clip_max = X_MAX
        self.cuda = CUDA
        self.clamp_fn = 'tanh'  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)

        loss2 = dist.sum()

        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        if self.clamp_fn == 'tanh':
            input_adv = tanh_rescale(modifier_var + input_var, self.clip_min, self.clip_max)
        else:
            input_adv = torch.clamp(modifier_var + input_var, self.clip_min, self.clip_max)

        output = model(input_adv)

        # distance to the original input data
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=False)

        loss = self._loss(output, target_var, dist, scale_const_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, input_adv_np

    def run(self, model, input, target, batch_idx=0):
        batch_size = input.size(0)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?
            input_var = autograd.Variable(torch_arctanh(input), requires_grad=False)
            input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max)
        else:
            input_var = autograd.Variable(input, requires_grad=False)
            input_orig = None

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            modifier = torch.normal(modifier, std=0.001)
        if self.cuda:
            modifier = modifier.cuda()
        modifier_var = autograd.Variable(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=LEARNING_RATE)

        for search_step in range(self.binary_search_steps):
            print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = autograd.Variable(scale_const_tensor, requires_grad=False)

            prev_loss = 1e6
            for step in range(self.max_steps):
                # perform the attack
                loss, dist, output, adv_img = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_orig)

                if step % 100 == 0 or step == self.max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(
                        step, loss, dist.mean(), modifier_var.data.mean()))

                if self.abort_early and step % (self.max_steps // 10) == 0:
                    # every 10% step check abort early
                    if loss > prev_loss * .9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    if di < o_best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]

                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack
        
def save_img(path,test_data_cln, test_data_adv, test_label, test_label_adv):
    imgpath = path + 'img_k{}_c{:.2f}/'.format(CONFIDENCE,INITIAL_CONST)
    #save adversarial example
    if Path(imgpath).exists()==False:
        Path(imgpath).mkdir(parents=True)
    toImg = transforms.ToPILImage()
    image = test_data_cln.cpu()
    image_adv = test_data_adv.cpu()
    label = test_label.cpu()
    label_adv = test_label_adv.cpu()
    tot = len(image)
    batch = 20
    for i in range(0, batch):
        #print(image[i].size())
        im = toImg(image[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_cln.jpg'.format(i,test_label[i])))
        im = toImg(image_adv[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_adv.jpg'.format(i,test_label_adv[i])))

def save_data_label(path, test_data_cln, test_data_adv, test_label, test_label_adv):
    if Path(path).exists() == False:
        Path(path).mkdir(parents=True)
    with open(Path(path)/Path('test_data_cln.p'),'wb') as f:
        pickle.dump(test_data_cln.cpu(), f, pickle.HIGHEST_PROTOCOL)

    with open(Path(path)/Path('test_adv(k{}c{:.3f}).p'.format(CONFIDENCE,INITIAL_CONST)),'wb') as f:
        pickle.dump(test_data_adv.cpu(), f, pickle.HIGHEST_PROTOCOL)

    with open(Path(path)/Path('test_label.p'),'wb') as f:
        pickle.dump(test_label.cpu(), f, pickle.HIGHEST_PROTOCOL)
    
    with open(Path(path)/Path('label_adv(k{}c{:.3f}).p'.format(CONFIDENCE,INITIAL_CONST)),'wb') as f:
        pickle.dump(test_label_adv.cpu(), f, pickle.HIGHEST_PROTOCOL)

def main():
    cw_attack = AttackCarliniWagnerL2()
    
    if MODEL == 'vgg':
        model = VGG.VGG16(enable_lat=False,
                          epsilon=0.5, 
                          pro_num=5,
                          batch_size=BATCHSIZE,
                          num_classes=NUM_CLASSES,
                          if_dropout=False)
    model.cuda()
    model.load_state_dict(torch.load((MODEL_PATH)))
    dataloader = return_data()
    # batch-norm and drop-out performs different in train() and eval()
    model.eval()
    cor_cln = 0
    cor_adv = 0
    tot = 0
    for step, (x,y) in enumerate(dataloader):
        print('step {}'.format(step))
        x = Variable(x, requires_grad = True).cuda()
        y_true = Variable(y, requires_grad = False).cuda()
        h = model(x)
        pred = torch.max(h,1)[1]
        cor_cln += (pred == y_true.data).sum().item()
        x_adv_np = cw_attack.run(model, x.detach(), y_true, step)
        x_adv = torch.from_numpy(x_adv_np).permute(0,3,1,2).cuda()
        print(type(x_adv),x_adv.size())
        h_adv = model(x_adv)
        pred_adv = torch.max(h_adv,1)[1]
        cor_adv += (pred_adv == y_true.data).sum().item()
        tot += y.size(0)
        print(x.data.size(),x_adv.data.size(),y.size())
        if step == 0:
            test_data_cln = x.data.detach()
            test_data_adv = x_adv.data
            test_label = y
            test_label_adv = pred_adv
        else:
            test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
            test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
            test_label = torch.cat([test_label, y],0)
            test_label_adv = torch.cat([test_label_adv, pred_adv],0)

    model.train()
    print("Before Carlini-L2 the accuracy is", float(100*cor_cln)/tot)
    print("After Carlini-L2 the accuracy is", float(100*cor_adv)/tot)
    return test_data_cln, test_data_adv, test_label, test_label_adv
if __name__ == '__main__':
    test_data_cln, test_data_adv, test_label, test_label_adv=main()
    PATH = "/media/dsg3/dsgprivate/lat/test_cw/"
    save_data_label(PATH, test_data_cln, test_data_adv, test_label, test_label_adv)
    save_img(PATH, test_data_cln, test_data_adv, test_label, test_label_adv)