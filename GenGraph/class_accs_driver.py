import os
import sys
import glob
import argparse
import numpy as np
from gen_graph import CIFAR_Net, convert_model_to_network,get_classwise_accuracy
import re

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l 
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_parser():
    #add command line arguments for easy training
    parser = argparse.ArgumentParser(description='ClasswiseAcc Driver')
    #note the default output_path is a new folder in the input paths
    parser.add_argument("--input_path",type=str,help='Specify Input directory',default='exps/exp_lr_0001_momentum_9_dr_1_bs_100_ne_100_sf_3_sgd_AUGDATA_sbmb_0')
    #default is to convert all in the directory
    args = parser.parse_args()

    print("=> Input Path: {}".format(args.input_path))
    return args


def main():
    args = arg_parser()
    glob_path = args.input_path + "/*.pth"
    print(glob_path)
    final_ckpt_file = sort_nicely(glob.glob(glob_path))[-1]
       
    net = CIFAR_Net()
    get_classwise_accuracy(net=net,ckpt_file=final_ckpt_file,save_path=args.input_path)
  
if __name__ == "__main__":
    main()