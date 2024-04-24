import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
import dice_models
import dataset
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
import sys

def read_ini(file_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    dataset_pkl = config["paths"]["dataset_pkl"]
    results_dir = config["paths"]["results_dir"]
    master_node = config["distributed"]["master"]
    port_num = config["distributed"]["port"]
    model = config["model"]["model"]
    epochs = config["model"]["epochs"]
    return dataset_pkl, results_dir, master_node, port_num, model, epochs

pklfile, dire, master_node, port_num, model, epochs = read_ini("/s/chopin/k/grad/mbrad/cs535/EEG_Classification/EEG-DICE-net/distributed_learning/config.ini")


def train(gpu, args):
    original_stdout = sys.stdout
    rank = args.nr * args.gpus + gpu
    print("Current GPU", gpu,"\n RANK: ",rank)

    torch.manual_seed(7)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    criterion = nn.BCEWithLogitsLoss()

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    maxAcc=0
    maxEpochs=0
    fold_accuracies=[]

    start = datetime.now()
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')
    for epoch in tqdm(range(epochs,epochs+5,10)):
    #########################################
    
    

        conv1outs=[]
        conv2outs=[]
        ## ROC CURVES
        all_y=[]
        all_probs=[]
        ####
        logo = LeaveOneGroupOut()
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],
                   'train_f1': [], 'test_f1': []}
        confusion_table=numpy.array([[0, 0], [0, 0]])
        cmatrixes=[]
        dataset = dataset.mlcDataset(path=pklfile)
        print(logo.get_n_splits(groups=dataset.subj))
        with warnings.catch_warnings():

            ###################################################################
            #STOP HERE:
            # Comment out one of the 2 following for loops
            # Choose if you want Leave-One-Group-Out OR K-FOLD
            
            #LEAVE ONE OUT
            for fold, (train_idx,val_idx) in enumerate(logo.split(X=numpy.arange(len(dataset)),y=dataset.y,groups=dataset.subj)):
            
            #KFOLD
            # for fold, (train_idx,val_idx) in enumerate(splits.split(numpy.arange(len(dataset)))):
                tqdm.write("FOLD: "+ str(fold))

            ###################################################################
                
                ###############################################################
                # STOP HERE:
                # Choose what model you want to run from dice_models.py
                ###############################################################
                ############################################
                criterion = torch.nn.BCEWithLogitsLoss()
                ############################################
                learning_rate = 0.001      # optimizer step
                weight_decay = 0.01        # L2 regularization
                optimizer_params = {'params': model.parameters(),
                                    'lr': learning_rate,
                                    'weight_decay': weight_decay}
                optimizer = torch.optim.AdamW(**optimizer_params)
                ############################################
                batch_size = 32    # batch size
                shuffle = True      # if you want to shuffle your data after each epoch
                drop_last = True   # the last batch (maybe) contains less samples than the batch_size maybe you do not want it's gradients
                num_workers = 0     # number of multi-processes data loading
                pin_memory = True   # enable fast data transfer to CUDA-enabled GPUs
                    # ---------- Properties ----------
                loader_params = {'batch_size': batch_size,
                                  'drop_last': drop_last,
                                  'num_workers': num_workers,
                                  'pin_memory': pin_memory}
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_idx,num_replicas=args.world_size,rank=rank)
                test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
                train_loader = torch.utils.data.DataLoader(dataset,sampler=train_sampler, **loader_params)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=test_sampler)
                
                for epoch in range(epochs): ##todo: everything below
                    train_loss, train_acc, train_f1, train_rec, train_pre = tef.train_epoch(train_loader, optimizer, model, criterion)
                    
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
                conv1,conv2=cm.get_weights(model)
                conv1outs.append(conv1)
                conv2outs.append(conv2)
                
        
        
        accuracy,sensitivity,specificity,precision,f1=cm.calc_scores_from_confusionmatrix(confusion_table)
        
        with open(dire+"/distributed_results_DICE-no-cnn_A-C.txt", 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print ("EPOCHS USED: ", epochs)
            print('accuracy',accuracy)
            print('f1_score',f1)
            print('sensitivity',sensitivity)
            print('specificity',specificity)
            print('precision',precision)
            sys.stdout = original_stdout # Reset the standard output to its original value

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser()
    # Master node may need to be an IP address, sardine is a lab machine at CSU.
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    print(f'initializing ddp: GLOBAL_RANK: {args.nr}, MEMBER: {int(args.nr)+1} / {args.nodes}')
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
    os.environ['MASTER_ADDR'] = master_node              #
    os.environ['MASTER_PORT'] = port_num                  #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################