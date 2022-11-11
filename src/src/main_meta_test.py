import os
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
# os.enviorn['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:20000'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import numpy as np
import sys
# import torch
from termcolor import colored
import pandas as pd
import collections

from train.train_utils import train_meta_function, train_function
from validation.val_utils import val_function, val_meta_function
# from input_configs import p
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
import collections
import time
import yaml
import argparse
import dill
import dill as pickle
from pytorchtools import EarlyStopping
from random import sample as SMP
from utils.meta_test_utils import * 
from utils.utils_common import *
# from utils.losses import ModelLossWrapper
parser = argparse.ArgumentParser(description='MetaTesting')
parser.add_argument('--config_exp', help='Config file for the meta test experiment')
args = parser.parse_args()

import warnings
warnings.filterwarnings('ignore')






def main():
    
    # get the config file
    p = create_config( args.config_exp) 
    
    meta_train_config = p['meta_train_folder_name']+'config_file.yaml'
    meta_train_p = create_config(meta_train_config)

    # merge dictionaries 
    p.update(meta_train_p)

    p['meta_train_task_list'] = p['task_list']

    for tasks in p['add_new_task_list']:
        p['task_list'].append(tasks)   # will be the new task list    

    print(p['task_list'])

    p['num_workers'] = 4 # just to cut down the cpu usage

    fname = '/proj/ltu_mtl/users/x_ricup/results/Test/'+ p['dataset_name']+'/additional_task_training/' +  'Test_' + p['Experiment_name_meta_test']
    if not os.path.exists(fname):
        os.makedirs(fname)
    else:
        print('folder already exist')

    #### save the configs for future use 
    with open(fname+'/'+'config_file.yaml','w')as file:
        doc = yaml.dump(p,file)
    
    writer = SummaryWriter(fname)

    print(colored('Retrieve model', 'blue'))

    # if 'segmentsemantic' in p['meta_train_task_list']: ## just for taskonomy because semantic segmentation tasks overfits easily 
    #     print('semantic seg. chk point')
    #     checkpoint = torch.load(p['meta_train_folder_name'] + 'segmentsemantic_checkpoint.pt')
    # else:
    checkpoint = torch.load(p['meta_train_folder_name'] + 'checkpoint.pt')
    model = checkpoint['model']

    ### freeze the model parameters 
    # if p['train_added_task_head_only'] == True:
    #     for params in model.parameters():
    #             params.requires_grad = False

    model_new = add_task_heads_to_model(model, p)
    model_new = model_new.cuda()

    
    print(colored('Get loss', 'blue'))
    loss_ft = get_criterion(p)
    loss_ft.cuda()
    print(loss_ft)

    # a wrapper for Multi-task learning using uncertainty to weigh losses
    complete_model_wrapper = ModelLossWrapper(p, loss_ft, model_new)

    # Get optimizer 
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, complete_model_wrapper)
    print(optimizer)


    # get scheduler 
    print(colored('Retrieve scheduler', 'blue'))
    scheduler = get_scheduler(p, optimizer)
    print(scheduler)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=p['earlystop_patience'], verbose=True, path=fname+'/'+ 'checkpoint.pt')    

    task_early_stopping = {}
    # initialize the task early_stopping object
    for task in p['task_list']:
        task_early_stopping[task] = EarlyStopping(patience=p['task_earlystop_patience'], verbose=True, path=fname+'/'+ task+'_checkpoint.pt')

    p['es_loss'] = {}   

    training_dict = {}
    validation_dict = {}      

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_dataset, val_dataset = get_dataset(p, fold=None)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    train_loader, val_loader = get_dataloader(p, train_dataset, val_dataset)

    start_epoch = 0     
    # best_loss = p['best_total_loss']   
    epochwise_train_losses_dict = collections.defaultdict(list)
    epochwise_val_losses_dict = collections.defaultdict(list)
    epochwise_train_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))   # for nested dictionary
    epochwise_val_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    
    train_time = 0

    inference_time = 0


    for epoch in range(start_epoch, p['epochs']):
        
        print(colored('Epoch %d/%d' %(epoch, p['epochs']-1), 'yellow'))
        print(colored('-'*10, 'yellow'))

        

        # if p['train_added_task_head_only'] == False:

        if (epoch >= p['finetuning_after']) :     
            for ct, child in enumerate(model_new.backbone.children()):        
                if ct == 7:
                    for param in child.parameters():
                        param.requires_grad = True
        elif (epoch >= 2* p['finetuning_after']): 
            for ct, child in enumerate(model_new.backbone.children()):        
                if ct >= 6:
                    for param in child.parameters():
                        param.requires_grad = True
        elif (epoch >= 3* p['finetuning_after']): 
            for ct, child in enumerate(model_new.backbone.children()):        
                if ct >= 5:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            for ct, child in enumerate(model_new.backbone.children()):   
                for param in child.parameters():
                    param.requires_grad = False            


        loss, metric,epoch_time = train_function(p, train_loader, model_new, complete_model_wrapper, epoch, writer,optimizer)


        # print('train loss: ', loss)
        # print('train metric: ', metric)


        train_time += epoch_time

        for keys in loss.keys():
            epochwise_train_losses_dict[keys].append(loss[keys])          
        
        for task,value in metric.items():
            for k,v in value.items():
                epochwise_train_metric_dict[task][k].append(v)
        

        tb_logger(p,epoch,loss,metric,writer,set='train')   #log on tensorboard


        vloss, vmetric,vepoch_time = val_function(p, val_loader, model_new, complete_model_wrapper, epoch, fname, scheduler, writer)
        # print('val loss: ', vloss)
        # print('val metric: ', vmetric)
        inference_time += vepoch_time            


        if p['setup'] == 'multi_task':
            for task, loss in vloss.items():
                if (task != 'total') and (p['flag'][task] == 1):
                    task_early_stopping[task](loss, model_new, epoch, optimizer, model_checkpoint=True)
                    if task_early_stopping[task].early_stop:
                        print("Early stopping task -", task)
                        p['flag'][task] = 0
                        p['es_loss'][task] = loss
                        head = model.decoders[task]
                        for params in head.parameters():
                            params.requires_grad = False

                elif (task != 'total') and (p['flag'][task] == 0):
                    vloss[task] = p['es_loss'][task]
                
        for keys in vloss.keys():
            epochwise_val_losses_dict[keys].append(vloss[keys])
        
        for task,value in vmetric.items():
            for k,v in value.items():
                epochwise_val_metric_dict[task][k].append(v)

        tb_logger(p,epoch,vloss,vmetric,writer,set='validation') #log on tensorboard   

        early_stopping(vloss['total'], model_new, epoch, optimizer, model_checkpoint=True)
        if (sum(p['flag'].values()) == 0 ) or (early_stopping.early_stop == True):
            print("Early stopping")
            break



        training_dict['train_time'] = train_time
        training_dict['loss'] = epochwise_train_losses_dict
        training_dict['metric'] = epochwise_train_metric_dict
        validation_dict['val_time'] = inference_time
        validation_dict['loss'] = epochwise_val_losses_dict
        validation_dict['metric'] = epochwise_val_metric_dict

        f = open(fname + '/'+ "training_dict.pkl","wb")
        dill.dump(training_dict,f)
        f.close()

        f = open(fname + '/'+ "validation_dict.pkl","wb")
        dill.dump(validation_dict,f)
        f.close()


    # ####### meta_testing on all the tasks ################

    # print(colored('Retrieve test_dataset', 'blue'))
    # test_dataset = get_test_dataset(p)
    # print('Test samples %d -' %(len(test_dataset)))
    # test_loader = get_test_dataloader(p, test_dataset)

    # meta_inference(p,test_loader,fname)




if __name__ == "__main__":
    main()

