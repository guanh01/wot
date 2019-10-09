
# coding: utf-8

import argparse
import os
import random
import shutil
import time, datetime 
import warnings
import sys
import numpy as np

import distiller 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from eval_util import AverageMeter, ProgressMeter, accuracy 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print(model_names)




parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/hguan2/datasets/imagenet/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False,
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--mode', default='WOT',
                    help='the training mode: QAT, ADMM, WOT')
parser.add_argument('--beta', default=0.0015, type=float,                  
                    help='beta for ADMM')
parser.add_argument('--debug', action='store_true', default=False,
                    help='if true, validate using only two batches. admm update for every iteration')
parser.add_argument('--logdir', default='./trained/',
                    help='the directory to store trained models and train logs.')

class Mode:
    QAT = 'QAT'
    ADMM = 'ADMM'
    WOT = 'WOT'
    



best_acc1 = 0


# use float32 as target accuracy 
target_accs_float = {
    "alexnet": 56.52,
    "squeezenet1_0": 58.09,
    "inception_v3": 69.54,
    "resnet152": 78.31,
    "resnet18": 69.76,
    "resnet34": 73.31,
    "resnet50": 76.13,
    "vgg16_bn": 73.36,
    "vgg16": 71.59,
}

# use int8 as target accuracy 
target_accs_int8 = {
    "alexnet": 55.8,
    "squeezenet1_0": 57.01,
    "inception_v3": 68.07,
    "resnet152": 77.79,
    "resnet18": 69.07,
    "resnet34": 72.83,
    "resnet50": 75.33,
    "vgg16_bn": 72.01,
    "vgg16": 71.51,
}

target_accs = target_accs_int8



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#             if i % args.print_freq == 0:
#                 progress.print(i)
            if i == 2 and args.debug:
                break 

    return top1.avg

def save_checkpoint(state, 
                    is_best):
    global time_string 
    filename = os.path.join(args.logdir, 
                            args.arch, 
                            args.mode,
                            time_string, 
                            'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        filedir = "/".join(filename.split('/')[:-1])
        shutil.copyfile(filename, os.path.join(filedir, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def clean_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('path already exist! remove path:', path)
    os.makedirs(path)

def check_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
def write_log(info):
    global time_string 
    filename = os.path.join(args.logdir, 
                            args.arch, 
                            args.mode,
                            time_string,
                            'train_details.log')
    filedir = "/".join(filename.split('/')[:-1])
    check_directory(filedir)
    with open(filename, 'a') as f:
        f.write(info)
        f.write('\n')

def check_large_weights_count(model):
    count = 0 # number of large weights 
    total = 0 # total number of weights 
    printed = False 
    for param_tensor in model.state_dict():
        name_part = param_tensor.split('.')
        if 'float_weight' in name_part:
            quantized_weight_name = param_tensor.replace('float_weight','')+'weight'
            quantized_weight_scale = param_tensor.replace('float_weight','')+'weight_scale'

            float_weight = model.state_dict()[param_tensor]

            weight = model.state_dict()[quantized_weight_name]
            weight_scale = model.state_dict()[quantized_weight_scale]

            
            weight_size = model.state_dict()[param_tensor].size()
            upper_bound = 63./weight_scale
            lower_bound = -64./weight_scale
            
            weight_flat = weight.view(-1)
            N = len(weight_flat)
            total += N
            
            if args.debug and not printed:
                printed = True 
                print('Wq: {}'.format(weight_flat[:5]))
                float_weight_flat = float_weight.view(-1)
                print('Wf: {}'.format(float_weight_flat[:5]))
                print('scale: {}'.format(weight_scale))

            
            change_idx_list_l = np.nonzero(weight_flat > upper_bound)
            change_idx_list_s = np.nonzero(weight_flat < lower_bound)
            change_idx_l_flat = change_idx_list_l.view(-1)
            change_idx_s_flat = change_idx_list_s.view(-1)
            overide_idx_l = np.nonzero(change_idx_l_flat % 8 != 7)
            overide_idx_s = np.nonzero(change_idx_s_flat % 8 != 7)

            count += overide_idx_l.nelement()
            count += overide_idx_s.nelement()
            
#     print('weight statistics: ', count, total, count/total)
    return count 


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    global model
    global optimizer
    
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    global train_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    global val_loader
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # add quantization scheduler
    global compression_scheduler
    compression_scheduler = distiller.CompressionScheduler(model) 
    compression_scheduler = distiller.file_config(
        model,
        optimizer, 
        '/home/hguan2/workspace/fault-tolerance/nips19/quant_aware_training.yaml', 
        compression_scheduler, 
        (args.start_epoch-1) if args.resume else None)
    
    model.cuda()
    global epoch
    
    info = "before training, large_weights_count: {}".format(check_large_weights_count(model))
    print(info)
    write_log(info)

    if args.mode == Mode.ADMM:
        global W_list, Z_list, U_list
        W_list = [] 
        Z_list = []
        U_list = [] 
        init_admm_param(model, W_list, Z_list, U_list)
        print("Initiaze ADMM parameters...")
        print("W_list:", W_list[0].view(-1)[:5])
        print("Z_list:", Z_list[0].view(-1)[:5])
        print("U_list:", U_list[0].view(-1)[:5])
    
    
    for epoch in range(args.start_epoch, args.epochs):
        compression_scheduler.on_epoch_begin(epoch)
    
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        regularized_train(compression_scheduler, train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        info = "epoch: {}, iteration: {}, after epoch, large_weights_count: {}, accuracy: {:.3f}".format(
                epoch,
                len(train_loader), 
                check_large_weights_count(model),
                acc1)
        write_log(info)
        print(info)
        

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        compression_scheduler.on_epoch_end(epoch,optimizer)
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            
        ## when it is clipping-based training, stop whenever the accuracy is met 
        if args.mode == Mode.WOT and best_acc1 >= target_accs[args.arch] + 1e-5:
            print("Reach best accuracy:{:.3f}, target_acc:{:.3f}".format(best_acc1, target_accs[args.arch]))
            return 


def init_admm_param(model, W_list, Z_list, U_list):
    # TODO: see if we want to use the quantized weight or float weight here for the regularization
    for param_tensor in model.state_dict():
        name_part = param_tensor.split('.')
        if 'float_weight' in name_part:
            quantized_weight_name = param_tensor.replace('float_weight','')+'weight'
            quantized_weight_scale = param_tensor.replace('float_weight','')+'weight_scale'
            
            weight = model.state_dict()[quantized_weight_name]
            weight_scale = model.state_dict()[quantized_weight_scale]
            W_list.append(weight)
            
            Z = weight.clone().detach()
            Z = projection(Z, weight_scale)
            Z_list.append(Z)
            U = torch.zeros_like(Z)
            U_list.append(U)

def update_admm_param(model, W_list, Z_list, U_list):
    # update Z and U 
    idx = 0
    W_Z_diff = 0
    Z_diff = 0
    U_norm = 0 
    for param_tensor in model.state_dict():
        name_part = param_tensor.split('.')
        if 'float_weight' in name_part:
            quantized_weight_name = param_tensor.replace('float_weight','')+'weight'
            quantized_weight_scale = param_tensor.replace('float_weight','')+'weight_scale'
            
            weight = model.state_dict()[quantized_weight_name]
            weight_scale = model.state_dict()[quantized_weight_scale]

            Z = weight + U_list[idx]
            Z = projection(Z, weight_scale)
            Z_diff += torch.norm(Z - Z_list[idx])
            W_Z_diff += torch.norm(weight - Z)
            
            Z_list[idx]= Z
            
            U = U_list[idx] + weight - Z
            U_list[idx] = U
            U_norm += torch.norm(U)
            idx += 1
#     print("W_list:", W_list[0].view(-1)[:5])
#     print("Z_list:", Z_list[0].view(-1)[:5])
#     print("U_list:", U_list[0].view(-1)[:5])
    return W_Z_diff/len(W_list), Z_diff, U_norm/len(W_list)
        
            
def projection(weight, weight_scale):
    target = [-64., 63.]
    upper_bound = target[1]/weight_scale
    lower_bound = target[0]/weight_scale
    
    weight_flat = weight.view(-1)
    change_idx_list_l = np.nonzero(weight_flat > upper_bound)
    change_idx_list_s = np.nonzero(weight_flat < lower_bound)
    change_idx_l_flat = change_idx_list_l.view(-1)
    change_idx_s_flat = change_idx_list_s.view(-1)
    overide_idx_l = np.nonzero(change_idx_l_flat % 8 != 7)
    overide_idx_s = np.nonzero(change_idx_s_flat % 8 != 7)
            
    weight_flat[change_idx_l_flat[overide_idx_l]] = upper_bound
    weight_flat[change_idx_s_flat[overide_idx_s]] = lower_bound

    return weight    

def admm_loss(output, target, criterion):
    loss = criterion(output, target)
    admm_loss = 0 
    for W, U, Z in zip(W_list, U_list, Z_list):
        admm_loss += 0.5 * args.beta * torch.norm(W - Z + U)
        del U 
        del Z 
    if args.debug:
        print('cross entroy loss:{}, admm loss:{}'.format(loss, admm_loss))
    return loss + admm_loss 



def regularized_train(compression_scheduler, train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    avg_time = time.time()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_time_total = time.time()
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # add quantization
        compression_scheduler.on_minibatch_begin(epoch, i, optimizer)
        # clipping weights 
        if args.mode == Mode.WOT: 
            regulate_quantized_weight(model)  
        
        # compute output
        output = model(input)
        if args.mode == Mode.ADMM:
            loss = admm_loss(output, target, criterion)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch, i, len(train_loader))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_time_total = time.time() - batch_time_total
        
        if i % args.print_freq == 0:
            progress.print(i)
        
        update_freq = int(len(train_loader)//5)
        if args.debug:
            update_freq = 2 
        if i%update_freq == 0:
#             print("evalute model before clipping...")
            acc1 = validate(val_loader, model, criterion, args)
            info = "epoch: {}, iteration: {}, before clipping, large_weights_count: {}, accuracy: {:.3f}, loss: {:.4f}".format(
                epoch,
                i, 
                check_large_weights_count(model),
                acc1,
                losses.avg)
            write_log(info)
            print(info)

            if args.mode == Mode.WOT: 
#                 print('evalaute model after clipping...')
                regulate_quantized_weight(model)
                acc1 = validate(val_loader, model, criterion, args)
                info = "epoch: {}, iteration: {}, after clipping, large_weights_count: {}, accuracy: {:.3f}, ".format(
                    epoch,
                    i, 
                    check_large_weights_count(model),
                    acc1)
                write_log(info)
                print(info)
                
            if args.mode == Mode.ADMM:
#                 print("update ADMM param...")
                W_Z_diff, Z_diff, U_norm = update_admm_param(model, W_list, Z_list, U_list)
                info = "epoch: {}, iteration: {}, update admm, W_Z_diff: {:.4f}, Z_diff: {:.4f}, U_norm: {:.4f}".format(
                    epoch,
                    i, 
                    W_Z_diff,
                    Z_diff, 
                    U_norm)
                write_log(info)
                print(info)
                
            ## when it is clipping-based training, stop whenever the accuracy is met 
            if args.mode == Mode.WOT and acc1 >= target_accs[args.arch] + 1e-5:
                print("Reach best accuracy:{:.3f}, target_acc:{:.3f}".format(acc1, target_accs[args.arch]))
                return 
        
            
# limit the quantized weight value to be -64 ~ 64. Downscale float weight accordingly
def regulate_quantized_weight(model):
    
    layer_id = 0
    for param_tensor in model.state_dict():
        name_part = param_tensor.split('.')
        if 'float_weight' in name_part:
            quantized_weight_name = param_tensor.replace('float_weight','')+'weight'
            quantized_weight_scale = param_tensor.replace('float_weight','')+'weight_scale'
        
            float_weight = model.state_dict()[param_tensor]
            weight = model.state_dict()[quantized_weight_name]
            weight_scale = model.state_dict()[quantized_weight_scale]

            # regulate the weight
            upper_bound = 63./weight_scale
            lower_bound = -64./weight_scale
            
            weight_flat = weight.view(-1)
            
            change_idx_list_l = np.nonzero(weight_flat > upper_bound)
            change_idx_list_s = np.nonzero(weight_flat < lower_bound)
            change_idx_l_flat = change_idx_list_l.view(-1)
            change_idx_s_flat = change_idx_list_s.view(-1)
            overide_idx_l = np.nonzero(change_idx_l_flat % 8 != 7)
            overide_idx_s = np.nonzero(change_idx_s_flat % 8 != 7)


            float_weight_flat = float_weight.view(-1)
            float_weight_flat[change_idx_l_flat[overide_idx_l]] = float_weight_flat[change_idx_l_flat[overide_idx_l]]*upper_bound/ weight_flat[change_idx_l_flat[overide_idx_l]]           
            float_weight_flat[change_idx_s_flat[overide_idx_s]] = float_weight_flat[change_idx_s_flat[overide_idx_s]]*lower_bound/ weight_flat[change_idx_s_flat[overide_idx_s]] 
            
            weight_flat[change_idx_l_flat[overide_idx_l]] = upper_bound
            weight_flat[change_idx_s_flat[overide_idx_s]] = lower_bound


args = parser.parse_args()
#args = parser.parse_args(["--pretrained", "--arch",  "resnet18", 
#                          "--batch-size", "128", "--gpu",  "0", '--mode', 'WOT'])

# create unique folder name 
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m%d%Y%H%M%S", named_tuple)

# print time 
info="\n\nstart a new train at time: {}".format(datetime.datetime.now())
write_log(info)
print(info)

# print arguments 
info = str(args)
write_log(info)
print(info)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
#         cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                  'disable data parallelism.')

if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

args.distributed = args.world_size > 1 or args.multiprocessing_distributed

ngpus_per_node = torch.cuda.device_count()
if args.multiprocessing_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
else:
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)

