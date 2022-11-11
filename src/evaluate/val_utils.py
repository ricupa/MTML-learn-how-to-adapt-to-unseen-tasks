import collections
import numpy as np
import torch
import os 
from utils.utils_common import evaluation_metrics,one_hot
from termcolor import colored


def val_function(p, val_loader, model, loss_wrapper, epoch, fname,scheduler):
    
    batch_losses_dict = collections.defaultdict(list)
    batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
    losses =  collections.defaultdict(list)  # for avg batch loss or epoch loss 
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # print(colored('val batch %d' %(i), 'yellow'))
            metric_dict = {}
            # Forward pass
            images = batch['image']
            targets = batch['targets']

            if 'segmentsemantic' in p['task_list']:
                targets['segmentsemantic'] = one_hot(targets['segmentsemantic'], class_num = 18)
                
            # image,targets = batch
            images = images.cuda()        
            
            output, loss_dict, log_params = loss_wrapper(images, targets)        

            metric_dict = evaluation_metrics(p, output, targets)
              
            # update metric values of every batch to a dictionary
            for task,value in metric_dict.items():
                for k,v in value.items():
                    batch_metric_dict[task][k].append(v)
            
            # update loss value of every batch to a dictionary
            for keys,value in loss_dict.items():
                batch_losses_dict[keys].append(value.detach().cpu().numpy())

    losses = {task: val.mean() for task, val in loss_dict.items()}    
    metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in batch_metric_dict.items()}
    scheduler.step(losses['total'])

    if losses['total'] < p['best_total_loss']:
        p['best_total_loss'] = losses['total']
        if not os.path.exists(fname+'/models'):
            os.makedirs(fname+'/models')
        torch.save(model.state_dict(), fname+'/models/'+'epoch_{}_loss_{}_model.pt'.format(epoch,p['best_total_loss']))


    return losses, metric