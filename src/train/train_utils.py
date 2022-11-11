import collections
import numpy as np
import torch
import os
from termcolor import colored
from utils.utils_common import evaluation_metrics, get_optimizer, get_criterion,one_hot
import time
from random import sample as SMP




def train_function(p, train_loader, model, loss_wrapper, epoch, writer, optimizer):
    
    batch_losses_dict = collections.defaultdict(list)
    batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
    # batch_metric_dict = {}
    losses =  collections.defaultdict(list)  # for avg batch loss or epoch loss 
    # prof.start()
    model.train()
    epoch_time = 0
    for i, batch in enumerate(train_loader):

        # print(colored('train batch %d' %(i), 'yellow'))
        batch_time_start = time.time()
        metric_dict = {}
        
        # Forward pass
        images = batch['image']
        targets = batch['targets']  

        

        images = images.cuda()       
        
        # images = torch.autograd.Variable(images)
        # targets = {task: torch.autograd.Variable(val) for task, val in targets.items()}
        targets = {task: val.cuda() for task, val in targets.items()}                     


        if p['missing_labels'] == True:       
            num = np.random.randint(1, int(100/p['percent_missing_labels']))       # generate a random number (int)        

            if num == 1:  # for int(100/self.percent_miss_labels) percent labels
                new_task_list = SMP(p['task_list'], len(p['task_list'])-1)  # just to miss one label 
                
                # targets[miss_task_label[0]] = None
            else:
                new_task_list = p['task_list']
        else:
            new_task_list = p['task_list']

        
        
        output, targets, loss_dict, weighted_task_loss = loss_wrapper(images, targets, new_task_list)    

    
        
        # Backward
        optimizer.zero_grad()
        backprop_loss = loss_dict['total']
        backprop_loss.backward()        
        optimizer.step()

        batch_time_stop = time.time()
        epoch_time += (batch_time_stop - batch_time_start)

        metric_dict = evaluation_metrics(p, output, targets) 
             

        # update metric values of every batch to a dictionary
        for task,value in metric_dict.items():
            for k,v in value.items():
                batch_metric_dict[task][k].append(v)

        # update loss value of every batch to a dictionary
        for keys, value in loss_dict.items():
            batch_losses_dict[keys].append(value.detach().cpu().numpy())
        
        
        # prof.step()


    # prof.stop()
    
    losses = {task: val.mean() for task, val in loss_dict.items()}  
    

    metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in batch_metric_dict.items()}
        
    return losses,metric,epoch_time





def train_meta_function(p, batch, model, loss_wrapper, epoch, writer, train_optimizer, task_combinations):
    
    batch_losses_dict = collections.defaultdict(list)
    batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))

    batch_params_list = collections.defaultdict(lambda: collections.defaultdict(list))
    # # batch_metric_dict = {}
    # losses =  collections.defaultdict(list)  # for avg batch loss or epoch loss 
    # prof.start()
    model.train()
    batch_time = 0

    batch_time_start = time.time()

    # epoch_time = 0
    for idx, new_task_list in enumerate(task_combinations):

        # print(colored('train batch %d' %(i), 'yellow'))        
        metric_dict = {}        
        # Forward pass
        images = batch['image']
        targets = batch['targets']          

        images = images.cuda() 

        targets = {task: val.cuda() for task, val in targets.items()}                
      
        
        output, targets, loss_dict, _ = loss_wrapper(images, targets, new_task_list)        
        
        # Backward
        train_optimizer.zero_grad()
        backprop_loss = loss_dict['total']
        backprop_loss.backward()
        train_optimizer.step()  

        batch_params_list[idx] = model.state_dict()     
        

        metric_dict = evaluation_metrics(p, output, targets)             

        # update metric values of every batch to a dictionary
        for task,value in metric_dict.items():
            for k,v in value.items():
                batch_metric_dict[task][k].append(v)

        # update loss value of every batch to a dictionary
        for keys, value in loss_dict.items():
            batch_losses_dict[keys].append(value.detach().cpu().numpy())
        
        
        # prof.step()

    batch_time_stop = time.time()
    batch_time += (batch_time_stop - batch_time_start)
    # prof.stop()
    
    losses = {task: val.mean() for task, val in loss_dict.items()}  
    

    metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in batch_metric_dict.items()}
        
    return losses, metric, batch_time, batch_params_list



