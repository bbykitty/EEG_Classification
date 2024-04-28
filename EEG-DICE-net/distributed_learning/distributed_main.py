import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
import dice_models
import dataset as ds
import platform
import copy
import numpy
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.metrics import confusion_matrix
import configparser
import pickle
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
import warnings
import train_eval_func as tef
import calc_metrics as cm
import dataset
import sys

def read_ini(file_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    dataset_pkl = config["paths"]["dataset_pkl"]
    results_file = config["paths"]["results_file"]
    master_node = config["distributed"]["master"]
    port_num = config["distributed"]["port"]
    model = config["model"]["model"]
    epochs = config["model"]["epochs"]
    return dataset_pkl, results_file, master_node, port_num, model, epochs


def train(gpu, args, pklfile, dire, use_model, epochs):
    original_stdout = sys.stdout
    rank = args.nr * args.gpus + gpu
    print("Current GPU", gpu,"\n RANK: ",rank)

    torch.manual_seed(7)

    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    fold_accuracies=[]

    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')
    epochs = int(epochs)
    for num_epochs in tqdm(range(epochs,epochs+5,10)):
    #########################################
    
        ## ROC CURVES
        all_y=[]
        all_probs=[]
        ####
        logo = LeaveOneGroupOut()
        confusion_table=numpy.array([[0, 0], [0, 0]])
        cmatrixes=[]
        dataset = ds.mlcDataset(path=pklfile)
        print(logo.get_n_splits(groups=dataset.subj))
        with warnings.catch_warnings():
            
            #LEAVE ONE OUT
            for fold, (train_idx,val_idx) in enumerate(logo.split(X=numpy.arange(len(dataset)),y=dataset.y,groups=dataset.subj)):
            
                tqdm.write("FOLD: "+ str(fold))
                ###############################################################
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = getattr(dice_models, use_model)()
                model = nn.parallel.DistributedDataParallel(model)
                model = model.to(device)
                ############################################
                criterion = torch.nn.BCEWithLogitsLoss()
                criterion.to(device)
                ############################################
                learning_rate = 0.001      # optimizer step
                weight_decay = 0.01        # L2 regularization
                optimizer_params = {'params': model.parameters(),
                                    'lr': learning_rate,
                                    'weight_decay': weight_decay}
                optimizer = torch.optim.AdamW(**optimizer_params)
                ############################################
                batch_size = 32    # batch size
                drop_last = True   # the last batch (maybe) contains less samples than the batch_size maybe you do not want it's gradients
                num_workers = 0     # number of multi-processes data loading
                pin_memory = True   # enable fast data transfer to CUDA-enabled GPUs
                masking = False
                    # ---------- Properties ----------
                loader_params = {'batch_size': batch_size,
                                  'drop_last': drop_last,
                                  'num_workers': num_workers,
                                  'pin_memory': pin_memory}
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_idx,num_replicas=args.world_size,rank=rank)
                test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
                train_loader = torch.utils.data.DataLoader(dataset,sampler=train_sampler, **loader_params)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=test_sampler)
                
                for epoch in range(epochs):
                    train_loss, train_acc, train_f1, train_rec, train_pre = tef.train_epoch(train_loader, optimizer, model, criterion,masking)
                    
                test_loss, test_acc, test_f1, test_rec, test_pre, conf_matrix = tef.eval_epoch(model, test_loader, criterion)
                y_true,y_pred_prob=tef.eval_epoch_ROC(model, test_loader, criterion)
                all_y.extend(y_true)
                all_probs.extend(y_pred_prob)
                print("test_acc: ",test_acc)
                
                ##calculate accuracies for each fold
                accuracy,_,_,_,_=cm.calc_scores_from_confusionmatrix(conf_matrix)
                fold_accuracies.append(accuracy)
                cmatrixes.append(conf_matrix)
                confusion_table = confusion_table + conf_matrix
                
        
        
        accuracy,sensitivity,specificity,precision,f1=cm.calc_scores_from_confusionmatrix(confusion_table)
        
        with open(dire, 'a') as f:
            sys.stdout = f
            print ("EPOCHS USED: ", epochs)
            print('accuracy',accuracy)
            print('f1_score',f1)
            print('sensitivity',sensitivity)
            print('specificity',specificity)
            print('precision',precision)
            sys.stdout = original_stdout

if __name__ == "__main__": 
    
    pklfile, dire, master_node, port_num, model, epochs = read_ini("/s/chopin/k/grad/mbrad/cs535/project/EEG_Classification/EEG-DICE-net/distributed_learning/config.ini")

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    args = parser.parse_args()
    print(f'initializing ddp: GLOBAL_RANK: {args.nr}, MEMBER: {int(args.nr)+1} / {args.nodes}')
    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
    os.environ['MASTER_ADDR'] = master_node
    os.environ['MASTER_PORT'] = port_num
    mp.spawn(train, nprocs=args.gpus, args=(args,pklfile, dire, model, epochs))
    #########################################################