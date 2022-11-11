import collections
import numpy as np
import torch
import os
from termcolor import colored
from utils.utils_common import evaluation_metrics, get_optimizer, get_criterion,one_hot,draw_segmentation_map_taskonomy,draw_segmentation_map_NYU,UnNormalize
import time
from random import sample as SMP
import torch.nn.functional as F
from random import sample as SMP
# from collections import Counter



def tb_visualization(images, targets, writer, epoch, p, task_list ):

    #### for tensorboard visulaization 
    img_idx = np.random.randint(low = 0, high = images.shape[0]) # for visualizing the results on tensorboard        
        # img_idx = 10        
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img = unorm(images[img_idx,:,:,:])              
    writer.add_image('Image', img, epoch)


        
    if 'segmentsemantic' in task_list:
            target = targets['segmentsemantic'][img_idx,0,:,:]
            # target = torch.squeeze(target,0)
            out = F.interpolate(output['segmentsemantic'], size=(256,256), mode="bilinear")
            out = F.softmax(out, dim = 1)
            out = out[img_idx,:,:,:]
            out = torch.argmax(out, dim = 0)

            if p['mode'] == 'binary':
                out = draw_binary_segmentation_map(p,out).permute(2,0,1)
                writer.add_image('segmentation_output', out, epoch)              
            else:
                if p['dataset_name'] == 'Taskonomy':
                    out = draw_segmentation_map_taskonomy(out).permute(2,0,1)
                    tar = draw_segmentation_map_taskonomy(target).permute(2,0,1)
                elif p['dataset_name'] == 'NYU':
                    out = draw_segmentation_map_NYU(out).permute(2,0,1)
                    tar = draw_segmentation_map_NYU(target).permute(2,0,1)
                else:
                    print('dataset not found')
                writer.add_image('Expected_seg_output', tar, epoch)
                writer.add_image('segmentation_output', out, epoch)         



    if 'depth_euclidean' in task_list:
        tar = F.interpolate(targets['depth_euclidean'], size=(256,256), mode="bilinear")
        tar = tar[img_idx,:,:,:]
        writer.add_image('depth_ground_truth', tar, epoch)  

        out = F.interpolate(output['depth_euclidean'], size=(256,256), mode="bilinear")
        out = out[img_idx,:,:,:]
        writer.add_image('depth_predicted_output', out, epoch)

    if  'edge_texture' in task_list:
            tar = F.interpolate(targets['edge_texture'], size=(256,256), mode="bilinear")
            tar = tar[img_idx,:,:,:]
            writer.add_image('edge_texture_truth', tar, epoch)
            # out = torch.sigmoid(output['edge_texture'])
            out = F.interpolate(output['edge_texture'], size=(256,256), mode="bilinear")
            out = out[img_idx,:,:,:]
            writer.add_image('edge_texture_output', out, epoch)

    if 'surface_normal' in task_list:
            tar = F.interpolate(targets['surface_normal'], size=(256,256), mode="bilinear")
            tar = tar[img_idx,:,:,:]
            writer.add_image('surface_normal_truth', tar, i)                    
            out = F.interpolate(output['surface_normal'], size=(256,256), mode="bilinear")
            out = out[img_idx,:,:,:]
            writer.add_image('surface_normal_output', out, i)

    return



def meta_training_function(p, train_loader, val_loader, model, complete_model_wrapper, epoch, writer, task_combinations,train_optimizer, meta_optimizer):
   
    tr_batch_losses_dict = collections.defaultdict(list)
    tr_batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
    val_batch_losses_dict = collections.defaultdict(list)
    val_batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
    
    losses =  collections.defaultdict(list)  # for avg batch loss or epoch loss 
    # prof.start()
    model.train()
    
    epoch_time = 0

    val_dataiter = iter(val_loader)


    for i, batch in enumerate(train_loader):

        if i == len(val_loader)-1:
            break    ### as validation batches are less than training batches 

        
        # print(colored('train batch %d' %(i), 'yellow'))
        batch_time_start = time.time()

        tr_images = batch['image']
        tr_targets = batch['targets']         

        tr_images = tr_images.cuda() 

        tr_targets = {task: val.cuda() for task, val in tr_targets.items()}

        val_batch = val_dataiter.next()
        val_images = val_batch['image'].cuda()
        val_targets = val_batch['targets']
        val_targets = {task: val.cuda() for task, val in val_targets.items()}


        ### for gradual fine tuning of the backbone        
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

        #### Initializations
        tr_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
        _tr_loss_dict = collections.defaultdict(list)
        val_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
        _val_loss_dict = collections.defaultdict(list)

        
        #### initialize gradient dictionary
        meta_grad_list = {}
        for i, params in enumerate(model.parameters()):            
            meta_grad_list[i] = torch.zeros_like(params)

        
        for idx, new_task_list in enumerate(task_combinations): 
            
            ###### for train set
            tr_output, tr_targets, tr_loss_dict, _ = complete_model_wrapper(tr_images, tr_targets, new_task_list) 

            ## backpropagate the total loss

            tr_bp_loss = torch.tensor(tr_loss_dict['total'], requires_grad= True)
            train_optimizer.zero_grad()
            tr_bp_loss.backward()
            train_optimizer.step()

            for keys, value in tr_loss_dict.items():
                _tr_loss_dict[keys].append(value.detach().cpu().numpy()) 
            
            tr_metrics = evaluation_metrics(p, tr_output, tr_targets) 

            for task,value in tr_metrics.items():
                for k,v in value.items():
                    tr_metric_dict[task][k].append(v)     

            #### for validation set
            val_output, val_targets, val_loss_dict, _ = complete_model_wrapper(val_images, val_targets, new_task_list)

            
            ### calculate gradients (torch.autograd.grad is not working)             
            val_loss_dict['total'].backward()     
            ### this will accumulate the gradients untill there is an optimizer step            


            for keys, value in val_loss_dict.items():
                _val_loss_dict[keys].append(value.detach().cpu().numpy())
            
            val_metrics = evaluation_metrics(p, tr_output, tr_targets) 

            for task,value in val_metrics.items():
                for k,v in value.items():
                    val_metric_dict[task][k].append(v)


        

        for params in model.parameters():
            if params.grad != None:
                params.grad = params.grad/len(task_combinations)

        meta_optimizer.step()
        meta_optimizer.zero_grad()

        tb_visualization(val_images, val_targets, writer, epoch, p, new_task_list)

        batch_time_stop = time.time()
        epoch_time += (batch_time_stop - batch_time_start)


        tr_loss_per_batch = {task: np.mean(val) for task, val in _tr_loss_dict.items()}  
        tr_metric_per_batch = {task: {m: np.mean(val) for m, val in values.items()} for task, values in tr_metric_dict.items()}

        val_loss_per_batch = {task: np.mean(val) for task, val in _val_loss_dict.items()}  
        val_metric_per_batch = {task: {m: np.mean(val) for m, val in values.items()} for task, values in val_metric_dict.items()}    


             

        # update train metric values of every batch to a dictionary
        for task,value in tr_metric_per_batch.items():
            for k,v in value.items():
                tr_batch_metric_dict[task][k].append(v)

        # update train loss value of every batch to a dictionary
        for keys, value in tr_loss_per_batch.items():
            tr_batch_losses_dict[keys].append(value)

        # update val metric values of every batch to a dictionary
        for task,value in val_metric_per_batch.items():
            for k,v in value.items():
                val_batch_metric_dict[task][k].append(v)

        # update val loss value of every batch to a dictionary
        for keys, value in val_loss_per_batch.items():
            val_batch_losses_dict[keys].append(value)
        
        
    
    train_losses = {task: np.mean(val) for task, val in tr_batch_losses_dict.items()}      

    train_metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in tr_batch_metric_dict.items()}


    validation_losses = {task: np.mean(val) for task, val in val_batch_losses_dict.items()}      

    validation_metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in val_batch_metric_dict.items()}
        
    return train_losses,train_metric,validation_losses,validation_metric,epoch_time