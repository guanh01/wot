# inject faults into weights 
import numpy as np 
import bitstring, time
from  collections import defaultdict  
import itertools, torch , collections 
from multiprocessing import Pool 
from functools import reduce
import bisect 

##############################################################
# fault injection: random bit flip 
##############################################################

def inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=False):
    """ For the input tensor, randomly choose n_bits number of bits to flip. 
    Total number of bits is num_values * 8.
    input tensor should be 1-d torch tensor. 
    Args:
    	- tensor (1-D torch tensor): the input torch tensor to inject faults. 
    	- random (numpy random object): the random object for generate random fault. 
    	- n_bits (int): the number of bits to flip. 
    	- debug_mode (bool): if true, print out some information for debugging. 
    Returns:
    	- stats (defaultdict): key-value pairs. The key is the index of a value whose bits are flipped. 
    	The value is a tuple: (value before flipped, flipped bit position, bit after flipped, value after flipped)"""
    
    assert len(tensor.size()) == 1, "The input tensor is not a 1-D vector. Current shape:{}".format(tensor.size())
    start = time.time()

    num_values = tensor.nelement()
    indexes = random.choice(num_values*8, size=n_bits, replace=False)
    sample_time = time.time() - start
    
    start = time.time() 
    stats = defaultdict(list)
    for index in indexes:
        vid, bid = index>>3, index&0b111
        value = int(tensor[vid])

        assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])

        bits = bitstring.pack('>b', value)

        bits[bid] ^= 1 
        value_after_flip = bits.int 

        tensor[vid] = value_after_flip   
        if debug_mode:
            print('vid: %5d, before: %5d, bid: %d => %s, after: %5d (%s)' 
                  %(vid, value, bid, bits[bid], value_after_flip, bits.bin)) 

        stats[vid].append((value, bid, bits[bid], value_after_flip))
    del indexes
    injection_time = time.time() - start
    print('sample time (s):', '%.4f' %(sample_time), ', injection_time (s):', '%.4f' %(injection_time)) 
    return stats 



##############################################################
# use parity bit to detect error and then set the value to zero 
##############################################################

def _parity_bit(v):
    bits = bitstring.pack('>b', v) # 8 bits
#     code = (sum(bits)%2 == 1)
    code = reduce(lambda x, y: x^y, bits)
    return code 
def _parity_bit_sum(v):
    bits = bitstring.pack('>b', v) # 8 bits
    code = (sum(bits)%2 == 1)
    return code 
def _parity_bit_numpy(v, width=8):
    bits = np.binary_repr(v, width=width) # 8 bits
    code = (sum(x=='1' for x in bits)%2 ==1)
    return code 

def _parity_bits(values):
    return [_parity_bit_numpy(int(v)) for v in values]


def factors(n):    
    return sorted(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
def _parity_encode(tensor):
    size = len(tensor)
    if size >= 128:
        fact = factors(size)
        index = bisect.bisect(fact, 128)
#         print('size:', size, ', factors:', fact, 'index:', index)
        split = fact[index]
#         print('split:', split) 
        tensor = tensor.view(split, -1)
        with Pool(32) as p:
            codes = p.map(_parity_bits, tensor)
        codes = torch.tensor(codes).view(-1)
    else:
        codes = torch.tensor(_parity_bits(tensor))
    return codes 

# def _parity_encode(tensor):
#     codes = [_parity_bit(v) for v in tensor]
#     with Pool() as p:
#         codes = p.map(_parity_bit, tensor)
#     return np.asarray(codes) 

def _correct_error_parity_zero(tensor, codes, flipped):
    new_codes = _parity_encode(tensor)
    # check when the two codes are different
    indexes = torch.nonzero(codes - new_codes)
    corr = {i.item(): (int(tensor[i].item()), 0, i.item() not in flipped) for i in indexes}
    tensor[indexes] = 0 
    return corr

def _correct_error_parity_avg(tensor, codes, flipped):
    new_codes = _parity_encode(tensor)
    # check when the two codes are different
    indexes = torch.nonzero(codes - new_codes)
    corr = {}
    for index in indexes:
        value = int(tensor[index].item())
        left = 0 if index == 0 else int(tensor[index-1].item())
        right = 0 if index == len(tensor)-1 else int(tensor[(index+1)%len(tensor)].item())
        new_value = int((left+right)/2)
        tensor[index] = new_value 
        corr[index.item()] = (value, new_value, index.item() not in flipped)
    return corr  

def inject_faults_int8_random_bit_position_parity_zero(tensor, random, n_bits, debug_mode=False):
    """ For the input tensor, apply parity zero protection. Input tensor should be 1-d torch tensor. 
    Total number of bits is num_values * 8. The procedure is: 
    Step 1: calculate parity encoding. 
    Step 2: randomly choose n_bits number of bits to flip for tensor 
    Step 3: randomly choose n_bits//8 number of bits to flip for the parity encoding. 
    Step 4: apply decoding based on parity codes. 
    The reason to seperatly flip tensor and parity codes is that we want to easily determine whether 
    a correction happens because of a faulty parity code or not. 

    Args:
    - tensor (1-D torch tensor): the input torch tensor to inject faults. 
    - random (numpy random object): the random object for generate random fault. 
    - n_bits (int): the number of bits to flip. 
    - debug_mode (bool): if true, print out some information for debugging. 
    Returns:
    - stats (dict): key-value pairs. The key is the index of a value whose bits are flipped. 
       The value is a tuple: (value before flipped, flipped bit position, bit after flipped, value after flipped)
    - corr (dict): key-value pairs. The key is the index of a value who is detected to be faulty based on parity encoding. 
       The value is a tuple: (value before correction, value after correction, False if parity bit is faulty) """
    assert len(tensor.size()) == 1, "The input tensor is not a 1-D vector. Current shape:{}".format(tensor.size())

    # step 1: parity encoding 
    start = time.time()
    codes = _parity_encode(tensor)
    if debug_mode:
        print('codes:', codes)
    encode_time = time.time() - start
    
    # step 2: inject faults to weights 
    stats = inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=debug_mode)
    
    # step 3: inject faults to codes
    start = time.time()
    num_values = tensor.nelement()
    indexes = random.choice(num_values, size=int(n_bits//8), replace=False)
    for i in indexes:
        codes[i] = 0 if codes[i]==1 else 1
    injection_time = time.time() - start 
    
    # step 4: error correction
    start = time.time()
    corr = _correct_error_parity_zero(tensor, codes, set(indexes))
    if debug_mode:
        print('stats:', sorted(stats.items()))
        print('faulty #codes:', len(indexes), sorted(indexes)) 
        print('correction:', sorted(corr.items()))
    correction_time = time.time() - start
    del indexes
    
    print('encoding time(s): %.4f, sample+injection time(s): %.4f, correction time(s): %.4f' 
          %( encode_time, injection_time, correction_time))
    return stats, corr 

def inject_faults_int8_random_bit_position_parity_avg(tensor, random, n_bits, debug_mode=False):
    assert len(tensor.size()) == 1, "The input tensor is not a 1-D vector. Current shape:{}".format(tensor.size())
    # step 1: parity encoding 
    start = time.time()
    codes = _parity_encode(tensor)
    if debug_mode:
        print('codes:', codes)
    encode_time = time.time() - start
    
    # step 2: inject faults to weights 
    stats = inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=debug_mode)
    
    # step 3: inject faults to codes
    start = time.time()
    num_values = tensor.nelement()
    indexes = random.choice(num_values, size=int(n_bits//8), replace=False)
    for i in indexes:
        codes[i] = 0 if codes[i]==1 else 1
    injection_time = time.time() - start 
    
    # step 4: error correction
    start = time.time()
    corr = _correct_error_parity_avg(tensor, codes, set(indexes))
    if debug_mode:
        print('faulty #codes:', len(indexes)) 
        print('correction:', corr)
    correction_time = time.time() - start
    del indexes
    
    print('encoding time(s): %.4f, sample+injection time(s): %.4f, correction time(s): %.4f' 
          %( encode_time, injection_time, correction_time))
    return stats, corr 


def _test_parity_bits():
    tensor = torch.randint(-40, 40, size=(100000,))
    
    s1 = time.time()
    t1 = torch.tensor(_parity_bits(tensor))
    e1 = time.time() - s1
    
    s2 = time.time()
    t2 = _parity_encode(tensor)
    e2 = time.time() - s2
    
    print(e1, e2, sum(t1-t2).item())

def _test_parity_bit():
    tensor = torch.randint(-40, 40, size=(5000,))
    s1 = time.time()
    t1 = [_parity_bit(v) for v in tensor]
    s1 = time.time() - s1
    
    s2 = time.time()
    t2 = [_parity_bit_sum(v) for v in tensor]
    s2 = time.time() - s2
    
    s3 = time.time()
    t3 = [_parity_bit_numpy(v) for v in tensor]
    s3 = time.time() - s3
    
    print(s1, s2, s3, sum(torch.tensor(t1) - torch.tensor(t2)).item(), sum(torch.tensor(t1) - torch.tensor(t3)).item())


##############################################################
# use SEC-DCD to detect error and correct error 
##############################################################
    
def _get_correctable_indexes(indexes, block_size=64, t=1):
    ''' 
    This method gets the bit indexes that can be corrected using ECC. 
    It tries to bucketize the indexes using block_size and check the number of indexes
    in each bucket. If the bucket contains no more than t number of indices, then these
    bits can be corrected via ECC.  TODO: This correction strategy is not accurate becuase 
    When #flips >= 3, Standard ECC is supposed to wrongly correct it. 
    This implementation ignore this case. 

    Args:
    	- indexes are bit indexes, 
    	- block_size (int): a data block is default 64. 
    	- t (int): #errors can be corrected. Default to be one.
    Return:
    	- corrected_indexes (set): the indices of bit positions that can be corrected. 
    '''
    corrected_indexes = set()
    blocks = defaultdict(list)
    for index in indexes:
        blocks[index//block_size].append(index)
    for block_id, block_faults in blocks.items():
        if len(block_faults) <= t:
            for k in block_faults:
                corrected_indexes.add(k)
    return corrected_indexes 

def inject_faults_int8_random_bit_position_bch(tensor, random, n_bits, debug_mode=False):
    return _inject_faults_int8_random_bit_position_ecc(tensor, random, n_bits, t=2, debug_mode=debug_mode)

def inject_faults_int8_random_bit_position_ecc(tensor, random, n_bits, debug_mode=False):
    return _inject_faults_int8_random_bit_position_ecc(tensor, random, n_bits, debug_mode=debug_mode)

def _inject_faults_int8_random_bit_position_ecc(
	tensor, 
	random, 
	n_bits, 
	block_size=64, 
	t=1, 
	debug_mode=False):
    
    '''
    For the input tensor, apply ecc protection. Input tensor should be 1-d torch tensor. 
    Total number of bits is num_values * 8. The procedure is: 
    Step 1: randomly choose n_bits number of bits to flip for tensor 
    Step 2: get the bit flipps that can be corrected via ECC. 
    Step 3: apply ECC correction.

    Args:
    - tensor: torch tensor
    - random: numpy random
    - n_bits: # bits to be flipped
    - block_size: the number of bits to be protected together to form a codeword
    - t: protection ability, can be 1 or 2 
    Returns:
    - stats (dict): key-value pairs. The key is the index of a value whose bits are flipped. 
    The value is a tuple: (value before flipped, flipped bit position, bit after flipped, value after flipped)
    - corr (dict): key-value pairs. The key is the index of a value who is detected to be faulty based on parity encoding. 
    The value is a tuple: (value before correction, value after correction)
    '''
    assert len(tensor.size()) == 1, "The input tensor is not a 1-D vector. Current shape:{}".format(tensor.size())
    # 1. fault injection with correction 
    start = time.time()
    num_values = tensor.nelement()
    indexes = random.choice(num_values*8, size=n_bits, replace=False)
    sample_time = time.time() - start
    
    # correct some error: put the indexes into data blocks, check whether a data block has more than two faults. 
    corrected_indexes = _get_correctable_indexes(indexes, block_size=block_size, t=t) 

    start = time.time() 
    stats = defaultdict(list)
    corr = {} 
    for index in indexes:
        vid, bid = index>>3, index&0b111
        value = int(tensor[vid])

        assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])

        bits = bitstring.pack('>b', value)
        bits[bid] ^= 1 
        value_after_flip = bits.int 
        
        # if the flip can be corrected:
        if index in corrected_indexes:
            corr[vid] = (value_after_flip, value)
        else:
            tensor[vid] = value_after_flip   
        
        if debug_mode:
            print('vid: %5d, before: %5d, bid: %d => %s, after: %5d (%s)' 
                  %(vid, value, bid, bits[bid], value_after_flip, bits.bin)) 

        stats[vid].append((value, bid, bits[bid], value_after_flip))
    injection_time = time.time() - start
    print('sample time (s):%.4f' %(sample_time),
              ', injection_time (s):%.4f' %(injection_time),
              ', #faults:%5d' %(n_bits),
              ', #corr:%5d' %(len(corr)))

    return stats, corr  

    


    
if __name__ == '__main__':
	pass 
    # _test_parity_bits()
    
