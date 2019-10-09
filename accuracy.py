
# coding: utf-8


import distiller 
import numpy as np
import os, collections, sys, shutil
import time 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision  
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib import pyplot as plt
from eval_util import test_imagenet   

import argparse

from datetime import datetime 
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print(model_names)

parser = argparse.ArgumentParser(description='CNN accuracy and weight distribution exploration')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--valdir', type=str, default='/home/hguan2/datasets/imagenet/val',
                    help='test dataset')
parser.add_argument('--dataset', type=str, default='imagenet', 
                    help='imagenet')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save pruned model (default: ./logs)')
parser.add_argument('--model-name', default='resnet18', type=str, 
                    help='architecture to use')
parser.add_argument('--num-batches', default=-1, type=int, 
                    help='number of batches used for testing')
parser.add_argument('--num-images-for-speed', default=500, type=int, 
                    help='number of images for testing inference speed')
# reconfigure 
parser.add_argument('--original-acc', action='store_true', default=False,
                    help='test the float32 model accuracy')
parser.add_argument('--quantized-acc', action='store_true', default=False,
                    help='test the 8-bit quantized model accuracy')
parser.add_argument('--w-dist', action='store_true', default=False,
                    help='print quantized model weight distribution')
parser.add_argument('--perch', action='store_true', default=False,
                    help='use per channel quant for weight quantization')

args = parser.parse_args() 
torch.manual_seed(args.seed)

print(args)


# ## helper functions

def clean_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('path already exist! remove path:', path)
    os.makedirs(path)

def check_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def print_weight_distribution(model):
    
    # check the distribution of parameters all weights
    thr = 32
    total_values, num_weights = 0, 0 
    counter = collections.Counter()
    for param_name, param in model.named_parameters():
        total_values += param.nelement()
        if len(param.size()) < 2:
            continue
        num_weights += param.nelement()
        counter.update(collections.Counter(np.abs(param.data.cpu().numpy().ravel())//thr + 1))

    tmp = sorted(counter.items(), key=lambda x: x[0])
    values, counts = zip(*tmp)
    
    # merge the interval [64, 96] and [96, 128]
    values = list(values)[:-1]
    counts = list(counts)
    counts[-2] += counts[-1]
    counts.pop() 

    total_weights = sum(list(counts))
    assert total_weights == num_weights
    print('#weights:', total_weights, ', #params:', total_values, 'percentage:', '%.6f' %(num_weights/total_values))
    
    percentages = [count*100/total_weights for count in counts]
    
    for interval, c, p in zip(['[0, 32)', '[32, 64)', '[64, 128]'], counts, percentages):
        print("{:>10}: {:>10d} {:>.2f}".format(interval, c, p))
        


# select model 
                    
print("=> using pre-trained model '{}'".format(args.model_name))
model = models.__dict__[args.model_name](pretrained=True)
model.cuda(args.gpu)

if args.quantized_acc or args.w_dist:
    print("prepare quantized model ...")
    dummy_input = torch.empty(1, 3, 224, 224)
    quantized_model = distiller.deepcopy(model)
    quantizer = distiller.quantization.PostTrainLinearQuantizer(
        quantized_model, 
        per_channel_wts=args.perch)
    quantizer.prepare_model(dummy_input)


if args.original_acc:
    print('test float model accuracy ...')
    s = time.time()
    acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
    s = time.time() - s
    print('Before quantization, accuracy: %.2f, time(s): %.2f' %(acc1, s))

if args.quantized_acc:
    print('test 8-bit quantized model accuracy ...')
    s = time.time() 
    acc1 = test_imagenet(quantized_model, args.valdir, num_batches=args.num_batches)
    s = time.time() - s 
    print('After quantization, accuracy: %.2f, time(s): %.2f' %(acc1, s))
    
if args.w_dist:
    print("print weight distribution ...")
    s = time.time() 
    print_weight_distribution(quantized_model)
    s = time.time() - s 
    print("time(s): %.2f" %(s))

