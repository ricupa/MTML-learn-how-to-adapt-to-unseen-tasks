import collections
import numpy as np
import torch
import os 
from utils.utils_common import evaluation_metrics,one_hot,draw_segmentation_map_taskonomy,draw_segmentation_map_NYU,UnNormalize, draw_binary_segmentation_map
from termcolor import colored
import torchvision
import time
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping

import torch.nn.functional as F

from random import sample as SMP


def val_function(p, val_loader, model, loss_wrapper, epoch, fname, scheduler, writer):    
    
    batch_losses_dict = collections.defaultdict(list)
    batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
    losses =  collections.defaultdict(list)  # for avg batch loss or epoch loss 
    
    model.eval()

    epoch_time = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            
            # print(colored('val batch %d' %(i), 'yellow'))
        
            metric_dict = {}
            batch_time_start = time.time()
            # Forward pass
            images = batch['image']
            targets = batch['targets']

            # if 'edge_texture' in p['task_list']:
            #     if targets['edge_texture'].numel() == 0
                
            # image,targets = batch
            images = images.cuda()      
            targets = {task: val.cuda() for task, val in targets.items()}  
            
            output, targets, loss_dict, weighted_task_loss = loss_wrapper(images, targets, p['task_list'])             

            batch_time_stop = time.time()
            epoch_time += (batch_time_stop - batch_time_start)

            metric_dict = evaluation_metrics(p, output, targets)           
            

            # update metric values of every batch to a dictionary
            for task,value in metric_dict.items():
                for k,v in value.items():
                    batch_metric_dict[task][k].append(v)
            
            # update loss value of every batch to a dictionary
            for keys,value in loss_dict.items():
                batch_losses_dict[keys].append(value.detach().cpu().numpy())
   
        #### for tensorboard visulaization 
        img_idx = np.random.randint(low = 0, high = images.shape[0]) # for visualizing the results on tensorboard        
        # img_idx = 10
        print(images.shape)
        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = unorm(images[img_idx,:,:,:])              
        writer.add_image('Image', img, epoch)

        
            
        if 'segmentsemantic' in p['task_list']:
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



        if 'depth_euclidean' in p['task_list']:
            tar = F.interpolate(targets['depth_euclidean'], size=(256,256), mode="bilinear")
            tar = tar[img_idx,:,:,:]
            writer.add_image('depth_ground_truth', tar, epoch)  

            out = F.interpolate(output['depth_euclidean'], size=(256,256), mode="bilinear")
            out = out[img_idx,0,:,:]
            out = out.unsqueeze(0)
            writer.add_image('depth_predicted_output', out, epoch)  


        if  'edge_texture' in p['task_list']:
            tar = F.interpolate(targets['edge_texture'], size=(256,256), mode="bilinear")
            tar = tar[img_idx,:,:,:]
            writer.add_image('edge_texture_truth', tar, epoch)
            # out = torch.sigmoid(output['edge_texture'])
            out = F.interpolate(output['edge_texture'], size=(256,256), mode="bilinear")
            out = out[img_idx,:,:,:]
            out = out/torch.max(out)
            writer.add_image('edge_texture_output', out, epoch)

        if 'surface_normal' in p['task_list']:
            tar = F.interpolate(targets['surface_normal'], size=(256,256), mode="bilinear")
            tar = tar[img_idx,:,:,:]
            writer.add_image('surface_normal_truth', tar, epoch)                    
            out = F.interpolate(output['surface_normal'], size=(256,256), mode="bilinear")
            out = out[img_idx,:,:,:]
            writer.add_image('surface_normal_output', out, epoch)



    losses = {task: val.mean() for task, val in loss_dict.items()}    
    metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in batch_metric_dict.items()}
    scheduler.step(losses['total'])

   
    # if losses['total'] < p['best_total_loss']:
    #     p['best_total_loss'] = losses['total']
    #     if not os.path.exists(fname+'/models'):
    #         os.makedirs(fname+'/models')            
    #         torch.save(model.state_dict(), fname+'/models/'+'epoch_{}_loss_{}_model.pt'.format(epoch,p['best_total_loss']))


    return losses, metric,epoch_time


##########################################################################################################################################
#########################################################################################################################


def val_meta_function(p, val_batch, model, loss_wrapper, epoch, fname, meta_optimizer,writer, task_combinations, batch_params_list):

    images = val_batch['image'].cuda()
    targets = val_batch['targets']
    targets = {task: val.cuda() for task, val in targets.items()}
    
    batch_losses_dict = collections.defaultdict(list)
    batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
    
    model.train()
    batch_time = 0
    batch_time_start = time.time()

    meta_optimizer.zero_grad()
    
    # with torch.no_grad():
    for idx, new_task_list in enumerate(task_combinations):
        # print(colored('val batch %d' %(i), 'yellow'))
        metric_dict = {}

        model.load_state_dict(batch_params_list[idx])  ### load parameters from the train function for the particluar task combination 
                
        output, targets, loss_dict, _ = loss_wrapper(images, targets, new_task_list)           

        
        backprop_loss = loss_dict['total']
        backprop_loss.backward()   ### accumulate the gradients    


        metric_dict = evaluation_metrics(p, output, targets)
            
        # update metric values of every batch to a dictionary
        for task,value in metric_dict.items():
            for k,v in value.items():
                batch_metric_dict[task][k].append(v)
        
        # update loss value of every batch to a dictionary
        for keys,value in loss_dict.items():
            batch_losses_dict[keys].append(value.detach().cpu().numpy())



    #### update the  meta gradients 
    meta_optimizer.step()    

    batch_time_stop = time.time()
    batch_time += (batch_time_stop - batch_time_start)


    losses = {task: val.mean() for task, val in loss_dict.items()}    
    metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in batch_metric_dict.items()}
    



    #### for tensorboard visulaization 
    img_idx = np.random.randint(low = 0, high = images.shape[0]) # for visualizing the results on tensorboard        
        # img_idx = 10        
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img = unorm(images[img_idx,:,:,:])              
    writer.add_image('Image', img, epoch)


        
    if 'segmentsemantic' in p['task_list']:
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



    if 'depth_euclidean' in p['task_list']:
        tar = F.interpolate(targets['depth_euclidean'], size=(256,256), mode="bilinear")
        tar = tar[img_idx,:,:,:]
        writer.add_image('depth_ground_truth', tar, epoch)  

        out = F.interpolate(output['depth_euclidean'], size=(256,256), mode="bilinear")
        out = out[img_idx,:,:,:]
        writer.add_image('depth_predicted_output', out, epoch)
        

    if  'edge_texture' in p['task_list']:
            tar = F.interpolate(targets['edge_texture'], size=(256,256), mode="bilinear")
            tar = tar[img_idx,:,:,:]
            writer.add_image('edge_texture_truth', tar, epoch)
            # out = torch.sigmoid(output['edge_texture'])
            out = F.interpolate(output['edge_texture'], size=(256,256), mode="bilinear")
            out = out[img_idx,:,:,:]
            writer.add_image('edge_texture_output', out, epoch)

    if 'surface_normal' in p['task_list']:
            tar = F.interpolate(targets['surface_normal'], size=(256,256), mode="bilinear")
            tar = tar[img_idx,:,:,:]
            writer.add_image('surface_normal_truth', tar, epoch)                    
            out = F.interpolate(output['surface_normal'], size=(256,256), mode="bilinear")
            out = out[img_idx,:,:,:]
            writer.add_image('surface_normal_output', out, epoch)



    return losses, metric, batch_time