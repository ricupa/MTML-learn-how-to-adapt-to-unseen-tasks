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
from utils.utils_common import *
from train.train_utils import train_meta_function
from validation.val_utils import val_meta_function
# from input_configs import p
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
import collections
import time
import yaml
import argparse
import dill as pickle
from pytorchtools import EarlyStopping
from random import sample as SMP


# from utils.losses import ModelLossWrapper
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()
import warnings
warnings.filterwarnings('ignore')


def main():
    
    # get the config file
    p = create_config( args.config_exp)    

    print(p['setup'])
    print(p['task_list'])

    fname = '../results/' +  p['dataset_name'] + '/' + p['Experiment_name']
    if not os.path.exists(fname):
        os.makedirs(fname)
    else: print('folder already exists')
    writer = SummaryWriter(fname)


    # Get model
    print(colored('Retrieve model', 'blue'))
    if p['checkpoint'] == True:
        print('checkpoint available')
        checkpoint = torch.load(p['checkpoint_folder'] + 'checkpoint.pt')
        model = checkpoint['model'] 
        start_epoch = checkpoint['epoch']+1
        with open(p['checkpoint_folder'] + 'validation_dict.pkl', 'rb') as f:
            validation_dict = pickle.load(f)
        with open(p['checkpoint_folder'] + 'training_dict.pkl', 'rb') as f:
            training_dict = pickle.load(f)      
        
        epochwise_train_losses_dict = training_dict['loss']
        epochwise_val_losses_dict = validation_dict['loss']
        epochwise_train_metric_dict = training_dict['metric']
        epochwise_val_metric_dict = validation_dict['metric']

    # Get model
    else:
        model = get_model(p)
        start_epoch = 0  
        epochwise_train_losses_dict = collections.defaultdict(list)
        epochwise_val_losses_dict = collections.defaultdict(list)
        epochwise_train_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))   # for nested dictionary
        epochwise_val_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    model = model.cuda()
    # Get criterion    
    loss_ft = get_criterion(p)
    loss_ft.cuda()
    print(loss_ft)    

    loss_wrapper = ModelLossWrapper(p, loss_ft, model)

    # Get optimizer 
    
    train_optimizer = get_optimizer(p, loss_wrapper)
    print('train_optimizer', train_optimizer)
    meta_optimizer = get_meta_optimizer(p, loss_wrapper)
    print('meta_optimizer',meta_optimizer)


    #### save the configs for future use 
    with open(fname+'/'+'config_file.yaml','w')as file:
        doc = yaml.dump(p,file)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=p['earlystop_patience'], verbose=True, path=fname+'/'+ 'checkpoint.pt')

    task_early_stopping = {}
    # initialize the task early_stopping object
    for task in p['task_list']:
        task_early_stopping[task] = EarlyStopping(patience=p['task_earlystop_patience'], verbose=True, path=fname+'/'+ task+'_checkpoint.pt')

    p['es_loss'] = {}
    ### find all the multi-task combinations
    task_combinations = get_combinations(p)   


    training_dict = {}
    validation_dict = {}      

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_dataset, val_dataset = get_meta_dataset(p, fold=None)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    train_loader, val_loader = get_meta_dataloader(p, train_dataset, val_dataset)

  
    epoch_time = 0

    for epoch in range(start_epoch, p['epochs']):
        
        print('Epoch %d/%d' %(epoch, p['epochs']-1))
        

        ### gradual fine-tuning of the model backbone
        if (p['pretrained'] == True) and  (p['finetuning'] == False):
            if (epoch >= p['finetuning_after']) :     
                for ct, child in enumerate(model.backbone.children()):        
                    if ct == 7:
                        for param in child.parameters():
                            param.requires_grad = True
            elif (epoch >= 2* p['finetuning_after']): 
                for ct, child in enumerate(model.backbone.children()):        
                    if ct >= 6:
                        for param in child.parameters():
                            param.requires_grad = True
            elif (epoch >= 3* p['finetuning_after']): 
                for ct, child in enumerate(model.backbone.children()):        
                    if ct >= 5:
                        for param in child.parameters():
                            param.requires_grad = True

            else:
                for ct, child in enumerate(model.backbone.children()):   
                    for param in child.parameters():
                        param.requires_grad = False


        ### initializations 

        tr_batch_losses_dict = collections.defaultdict(list)
        tr_batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))

        val_batch_losses_dict = collections.defaultdict(list)
        val_batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))

        val_dataiter = iter(val_loader)
        batch_time = 0

        for i, batch in enumerate(train_loader):

            if i == len(val_loader)-1:
                break    ### as validation batches are less than training batches, want to work for equal num_batches for train and val

            losses, metric, tr_batch_time, batch_params_list = train_meta_function(p, batch, model, loss_wrapper, epoch, writer, train_optimizer, task_combinations)          


            val_batch = val_dataiter.next()

            vlosses, vmetric, v_batch_time = val_meta_function(p, val_batch, model, loss_wrapper, epoch, fname, meta_optimizer,writer, task_combinations, batch_params_list)

            ########### updating loss and metric dictionaries 

            # update metric values of every batch to a dictionary
            for task,value in metric.items():
                for k,v in value.items():
                    tr_batch_metric_dict[task][k].append(v)

            # update loss value of every batch to a dictionary
            for keys, value in losses.items():
                tr_batch_losses_dict[keys].append(value.detach().cpu().numpy())

            # update metric values of every batch to a dictionary
            for task,value in vmetric.items():
                for k,v in value.items():
                    val_batch_metric_dict[task][k].append(v)

            # update loss value of every batch to a dictionary
            for keys, value in vlosses.items():
                val_batch_losses_dict[keys].append(value.detach().cpu().numpy())
            

            batch_time += (tr_batch_time + v_batch_time)

        
        epoch_time += batch_time

        

        tr_batchlosses = {task: np.mean(val) for task, val in tr_batch_losses_dict.items()}    
        tr_batchmetric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in tr_batch_metric_dict.items()}

        val_batchlosses = {task: np.mean(val) for task, val in val_batch_losses_dict.items()}    
        val_batchmetric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in val_batch_metric_dict.items()}


        tb_logger(p,epoch,tr_batchlosses,tr_batchmetric,writer,set='train')   #log on tensorboard

        

        for keys in tr_batchlosses.keys():
            epochwise_train_losses_dict[keys].append(tr_batchlosses[keys])          
        
        for task,value in tr_batchmetric.items():
            for k,v in value.items():
                epochwise_train_metric_dict[task][k].append(v)
        

        if p['setup'] == 'multi_task':
            for task, loss in val_batchlosses.items():
                if (task != 'total') and (p['flag'][task] == 1):
                    task_early_stopping[task](loss, model, epoch, train_optimizer, model_checkpoint=True)
                    if task_early_stopping[task].early_stop:
                        print("Early stopping task -", task)
                        p['flag'][task] = 0
                        p['es_loss'][task] = loss
                        head = model.decoders[task]
                        for params in head.parameters():
                            params.requires_grad = False

                elif (task != 'total') and (p['flag'][task] == 0):
                    val_batchlosses[task] = p['es_loss'][task]

        tb_logger(p,epoch,val_batchlosses,val_batchmetric,writer,set='validation') #log on tensorboard

        for keys in val_batchlosses.keys():
                epochwise_val_losses_dict[keys].append(val_batchlosses[keys])
            
        for task,value in val_batchmetric.items():
            for k,v in value.items():
                epochwise_val_metric_dict[task][k].append(v)       
        
        
           
        early_stopping(val_batchlosses['total'], model, epoch, train_optimizer, model_checkpoint=True)
        if (sum(p['flag'].values()) == 0 ) or (early_stopping.early_stop == True):
            print("Early stopping")
            break

       
        training_dict['train_time'] = epoch_time
        training_dict['loss'] = epochwise_train_losses_dict
        training_dict['metric'] = epochwise_train_metric_dict
        
        validation_dict['loss'] = epochwise_val_losses_dict
        validation_dict['metric'] = epochwise_val_metric_dict

        f = open(fname + '/'+ "training_dict.pkl","wb")
        pickle.dump(training_dict,f)
        f.close()

        f = open(fname + '/'+ "validation_dict.pkl","wb")
        pickle.dump(validation_dict,f)
        f.close()





if __name__ == "__main__":
    main()

