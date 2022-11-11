import os
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
# os.enviorn['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:20000'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "5"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import numpy as np
import sys
# import torch
from termcolor import colored
import pandas as pd
import collections
from utils.utils_common import *
from train.train_utils import train_function
from validation.val_utils import val_function
# from input_configs import p
from torch.utils.tensorboard import SummaryWriter
# import torch.profiler
import collections
import time
import yaml
import argparse
import dill as pickle
import warnings
warnings.filterwarnings('ignore')

from torchsummary import summary


from pytorchtools import EarlyStopping
# from utils.losses import ModelLossWrapper
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()


def main():

    global p
    # get the config file
    p = create_config( args.config_exp) 

    # print(colored(p['setup'], 'blue'))
    # print(colored(p['task_list'], 'blue'))
    #print dataset name 
    dataset_name = p['dataset_name']
    # print(colored(dataset_name, 'blue'))    


    fname = '../results/' + p['dataset_name'] + '/' + p['Experiment_name']
    if not os.path.exists(fname):
        os.makedirs(fname)
    else:
        print('folder already exist')
        
    writer = SummaryWriter(fname)

    # print(colored('Retrieve model', 'blue'))

    if p['checkpoint'] == True:
        # print('checkpoint available')
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

    loss_ft = get_criterion(p)
    loss_ft.cuda()
    # print(loss_ft)    

    # a wrapper for Multi-task learning using uncertainty to weigh losses
    complete_model_wrapper = ModelLossWrapper(p, loss_ft, model)

    # Get optimizer 
    # print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, complete_model_wrapper)
    # print(optimizer)

    # get scheduler 
    # print(colored('Retrieve scheduler', 'blue'))
    scheduler = get_scheduler(p, optimizer)
    # print(scheduler)

    

    #### save the configs for future use 
    with open(fname+'/'+'config_file.yaml','w')as file:
        doc = yaml.dump(p,file)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=p['earlystop_patience'], verbose=True, path=fname+'/'+ 'checkpoint.pt')
    

    task_early_stopping = {}
    # initialize the task early_stopping object
    for task in p['task_list']:
        task_early_stopping[task] = EarlyStopping(patience=p['task_earlystop_patience'], verbose=True, path=fname+'/'+ task+'_checkpoint.pt')

    p['es_loss'] = {} ## temperory for detecting the early stopping

    training_dict = {}
    validation_dict = {}      

    # Dataset
    # print(colored('Retrieve dataset', 'blue'))
    train_dataset, val_dataset = get_dataset(p, fold=None)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    train_loader, val_loader = get_dataloader(p, train_dataset, val_dataset)
    
    train_time = 0

    inference_time = 0

    for epoch in range(start_epoch, p['epochs']):
        
        print('Epoch %d/%d' %(epoch, p['epochs']-1))
        print(colored('-'*10, 'yellow'))

        #### for gradual fine tuning of the model's backbone
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


        loss, metric,epoch_time = train_function(p, train_loader, model, complete_model_wrapper, epoch, writer,optimizer)

        train_time += epoch_time

        for keys in loss.keys():
            epochwise_train_losses_dict[keys].append(loss[keys])          
        
        for task,value in metric.items():
            for k,v in value.items():
                epochwise_train_metric_dict[task][k].append(v)
        

        tb_logger(p,epoch,loss,metric,writer,set='train')   #log on tensorboard

        if 'eval_every_10th_epoch' in p.keys() and p['eval_every_10th_epoch']: 
            if epoch%10 == 0:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = True

        if eval_bool:
            vloss, vmetric,vepoch_time = val_function(p, val_loader, model, complete_model_wrapper, epoch, fname, scheduler, writer)
            print('val loss: ', vloss)
            print('val metric: ', vmetric)
            inference_time += vepoch_time            


            if p['setup'] == 'multi_task':
                for task, loss in vloss.items():
                    if (task != 'total') and (p['flag'][task] == 1):
                        task_early_stopping[task](loss, model, epoch, optimizer, model_checkpoint=True)
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

            early_stopping(vloss['total'], model, epoch, optimizer, model_checkpoint=True)
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
        pickle.dump(training_dict,f)
        f.close()

        f = open(fname + '/'+ "validation_dict.pkl","wb")
        pickle.dump(validation_dict,f)
        f.close()




if __name__ == "__main__":
    main()

