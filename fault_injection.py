
import argparse
import numpy as np
import os, shutil 

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision 
import torchvision.models as models
 
import pickle, time, collections
from datetime import datetime  
import distiller 
from eval_util import test_imagenet 
from eval_util import AverageMeter, ProgressMeter, accuracy 
from fault_util import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print(model_names)

# settings
parser = argparse.ArgumentParser(description='Fault Injection')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dataset', type=str, default='imagenet', 
                    help='imagenet')
parser.add_argument('--valdir', type=str, default='/home/hguan2/datasets/imagenet/val',
                    help='test dataset')
parser.add_argument('--save', default='./sim', type=str, metavar='PATH',
                    help='path to save simulation results (default: none)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# reconfigure 
parser.add_argument('--arch', default='vgg16', type=str, 
                    help='architecture to use')
parser.add_argument('--num-batches', type=int, default=10,
                    help='number of batches used for testing accuracy. -1 means using all batches.')
parser.add_argument('--fault-type', default='faulty', type=str,
                    help='fault type: {faulty, zero, avg, ecc, inplace}')
parser.add_argument('--start-trial-id', type=int, default=0,
                    help='start trial id')
parser.add_argument('--end-trial-id', type=int, default=1,
                    help='end trial id (not included)')
parser.add_argument('--clean-dir', action='store_true', default=False,
                    help='clean directory')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='the QAT-trained model')


args = parser.parse_args()
torch.manual_seed(args.seed)

args.save = os.path.join(args.save, args.arch, args.dataset, args.fault_type) 
if os.path.exists(args.save):
    if args.clean_dir:
        shutil.rmtree(args.save)
        print('path already exist! remove path:', args.save)
else:
    os.makedirs(args.save)
print('log will save to:', args.save)
print(args)

def select_fault_injection_function():
    fn = {
              'faulty': inject_faults_int8_random_bit_position, 
              'zero': inject_faults_int8_random_bit_position_parity_zero,
              'avg': inject_faults_int8_random_bit_position_parity_avg,
              'ecc': inject_faults_int8_random_bit_position_ecc,
              'inplace': inject_faults_int8_random_bit_position_ecc,
              'bch': inject_faults_int8_random_bit_position_bch,

         }
    assert args.fault_type in fn, "fault type: {} is not supported".format(args.fault_type)

    return fn[args.fault_type]  



def check_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def save_pickle(save_path, save_name, save_object):
    check_directory(save_path)
    filepath = os.path.join(save_path, save_name)
    pickle.dump(save_object, open(filepath,"wb" ))
    print('File saved to:', filepath)

def load_pickle(load_path, load_name=None, verbose=False):
    if load_name:
        filepath =  os.path.join(load_path, load_name)
    else:
        filepath = load_path 
    if verbose:
        print('Load pickle file:', filepath)
    return pickle.load( open(filepath, "rb" ))

def load_checkpoint(model_path):

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        best_prec1 = checkpoint['best_acc1']
        # checkpoint state_dict:
        # for var_name in checkpoint['state_dict']:
        # 	print(var_name, checkpoint['state_dict'][var_name].size())
        # for var_name in model.state_dict():
        # 	print(var_name, model.state_dict()[var_name].size())
        print('model state_dict len:', len(model.state_dict()))
        print("checkpoint state_dict len:", len(checkpoint['state_dict']))
        assert len(model.state_dict().keys() - checkpoint['state_dict'].keys())==0, "model vars should be inside checkpoint"
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}' Prec1: {:f}"
          .format(model_path, best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(model_path))
    return best_prec1 



def quantize_model_(model):
    # use the default setting, change model inplace 
    # https://github.com/NervanaSystems/distiller/blob/master/distiller/quantization/range_linear.py#L573
    # __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32, bits_overrides=None,
    #            mode=LinearQuantMode.SYMMETRIC, clip_acts=False, no_clip_layers=None, per_channel_wts=False,
    #            model_activation_stats=None):
    dummy_input = torch.empty(1, 3, 224, 224)
    quantizer = distiller.quantization.PostTrainLinearQuantizer(model)
    quantizer.prepare_model(dummy_input)
    
    # test the accuracy of the quantized model
    save_path = args.save # "/".join(args.save.split('/')[:-1])   
    num_batches = args.num_batches 
    model.cuda(args.gpu)
    prec1 = test_imagenet(model, args.valdir, num_batches=num_batches)
    # write the accuracy 
    with open(os.path.join(save_path, "quantize.txt"), "w") as fp:
        fp.write("Test accuracy: %.2f, num_batches: %d\n" %(prec1, num_batches)) 
  


def write_detailed_info(log_path, info):
    with open(os.path.join(log_path, 'logs.txt'), 'a') as f:
        f.write(info+'\n')
        
def get_weights(model):
    # get weights stat    
    weights = [] 
    weights_names = [] 
    for name, param in model.named_parameters():
        # don't do simulation on bias and batch normalization layer
        if len(param.size()) >= 2:
            weights.append(param) 
            weights_names.append(name)
    return weights, weights_names
    
        
def perturb_weights(model, n_faults, trial_id, log_path, fault_injection_fn): 
    # use trial_id to setup random seed 
    start = time.time()
    np.random.seed(trial_id)
    random = np.random  
    flipped_bits, changed_params, stats = 0, 0, {}
    
    # get the n_bits for each weight 
    weights, _ = get_weights(model)
    weights_sizes = [param.nelement() for param in weights]
    total_values = sum(weights_sizes)
    p = [size/total_values for size in weights_sizes]
    samples = random.choice(len(weights), size=n_faults, p=p)
    counter = collections.Counter(samples)
    
    print('samples:', sorted(counter.items())) 
    
    for weight_id in sorted(counter.keys()):
        param = weights[weight_id]
        tensor = param.data.view(-1)
#         tensor_copy = tensor.clone() 
        
        # flip n_bits number of values from tensor
        n_bits = counter[weight_id]
        res = fault_injection_fn(tensor, random, n_bits)
        stats[weight_id] = res 
        
        if isinstance(res, tuple):
            flipped_bits += sum([len(arr) for x, arr in stats[weight_id][0].items()])
            changed_params += len(stats[weight_id][0])
#             print('nonzero', torch.nonzero(tensor_copy.view(-1) - tensor.view(-1)).size()[0], len(stats[weight_id][0]))
        else:
            flipped_bits += sum([len(arr) for x, arr in stats[weight_id].items()])
            changed_params += len(stats[weight_id])
#             print('nonzero', torch.nonzero(tensor_copy.view(-1) - tensor.view(-1)).size()[0], len(stats[weight_id]))
    
    assert flipped_bits == n_faults and changed_params <= n_faults, '%d, %d, %d' %(flipped_bits, changed_params, n_faults) 
    
    total_bits = total_values* 8
    info = 'trial: %d, n_faults: %d, total_params: %d' %(trial_id, n_faults, total_values)
    info += ', flipped_bits: %d (%.2e)' %(flipped_bits, flipped_bits*1.0/total_bits)
    info += ', changed_params: %d (%.2e)' %(changed_params, changed_params*1.0/total_values)
    
    end = time.time() - start
    print('Finish fault injection, time (s):', end) 
    
    save_path = os.path.join(log_path, 'stats')
    save_name = str(trial_id) + '.pkl'
    save_pickle(save_path, save_name, stats)
    
    return info  

        
# select fault injection mode 
fault_injection_fn = select_fault_injection_function()

# select model 
if args.fault_type == 'inplace':
    model = models.__dict__[args.arch](pretrained=False)
    load_checkpoint(args.checkpoint)
else:
    model = models.__dict__[args.arch](pretrained=True)

# quantize model and keep a fault-free copy 
quantize_model_(model)
model_copy = distiller.deepcopy(model)

# collect weight stats 
weights, weights_names = get_weights(model)
weights_sizes = [param.nelement() for param in weights]
total_values = sum(weights_sizes)
print('# weights params:', len(weights), ', total_values:', total_values)
for i, item in enumerate(zip(weights_names, weights_sizes)):
    print('\t', i, item[0], item[1], '(%f)' %(item[1]/total_values))

##########################
## start simulation ######
##########################
# for each fault_rate, use fault rate to get the number of faults

print('\nSimulation start: ', datetime.now())
simulation_start = time.time()
# fault_rates = [10**x for x in range(-9, -2, 1)]
fault_rates = [0.0001] 
for fault_rate in fault_rates:
    
    n_faults = int(total_values * 8 * fault_rate)
    if n_faults <= 0: 
        continue
    
    folder = 'r%s' %(fault_rate)
    log_path = os.path.join(args.save, folder)
    
    # for each trial, initialize the model  
    for trial_id in range(args.start_trial_id, args.end_trial_id):
        print('\nfault_rate:', fault_rate, ', n_faults:', n_faults, ', trial_id:', trial_id)
        start = time.time()
        
        model = distiller.deepcopy(model_copy)
        model.cpu()
#         tensor_before = list(model.parameters())[0].clone()
        
        info = perturb_weights(model, n_faults, trial_id, log_path, fault_injection_fn) 

        model.cuda(args.gpu)
        acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
        
#         tensor_after = list(model.parameters())[0].cpu()
#         print('tensor_after - tensor_before', torch.nonzero(tensor_after - tensor_before).size())

        duration = time.time() - start  

        info += ', test_time: %d' %(duration)
        info += ', test_accuracy: %f' %(acc1)
        print(info, '\n')
        write_detailed_info(log_path, info)

 
simulation_time = time.time() - simulation_start
print('Simulation ends:', datetime.now(), ', duration(s):%.2f' %(simulation_time)) 
                 
    

    
    



        
    


        

        
        



    

