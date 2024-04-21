# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:28:13 2023

@author: AndreasMiltiadous
"""

import torch
import os
import numpy
import random

import pickle

def reproducability(seed):
    torch.backends.cudnn.deterministic = True      # initialize cuda backend
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)               # if you are using multi-GPU.
    numpy.random.seed(seed)                        # numpy random seed
    random.seed(seed)                              # random generator random seed
    torch.manual_seed(seed)                        # torch seed
    torch.cuda.manual_seed(seed)                   # torch cudda seed

def create_pickle_fpr_tpr(fpr,tpr,dir,filename="test"):
    with open(dir+"\\"+filename+'.pkl', 'wb') as f:
        pickle.dump(fpr, f)
        pickle.dump(tpr, f)
        
    
def create_pickle_accuracy(acc_list,dir,filename="accuracies"):
    with open(dir+"\\"+filename+'.pkl', 'wb') as f:
        pickle.dump(acc_list, f)