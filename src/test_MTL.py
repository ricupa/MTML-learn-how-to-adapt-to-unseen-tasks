
### python test_MTL.py --test_exp /proj/ltu_mtl/users/x_ricup/results/Test/NYU/additional_task_training/Test_8_2_meta_multi_add_edge_seg_depth_surface_trial_2_lessdata_50/

###  python test_MTL.py --test_exp /proj/ltu_mtl/users/x_ricup/results/NYU/8_3_meta_multi_seg_depth_surface_edge_trial_3/

### conda activate MTLenv
### python test_MTL.py --test_exp /proj/ltu_mtl/users/x_ricup/results/Taskonomy/8_3_meta_multi_seg_depth_surface_edge_trial_3/


### python test_MTL.py --test_exp /proj/ltu_mtl/users/x_ricup/results/Test/Taskonomy/additional_task_training/Test_10_2_meta_multi_add_depth_seg_surface_edge_trial_3/


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
from torchvision import transforms
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
import cv2
from torchsummary import summary
import seaborn as sns


parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--test_exp', help='path of folder to the config file  and checkpoint for testing')
args = parser.parse_args()

from decimal import *
getcontext().prec = 4






def main():

    dir = args.test_exp 
    
    config_file_name = dir + 'config_file.yaml'
    global p
    p = create_config( config_file_name)
    p['train_dir'] = dir

    if 'Experiment_name_meta_test' in p.keys():
        print('Exp name = ', p['Experiment_name_meta_test'])
    else:
        print('Exp name = ',p['Experiment_name'])

    print(p['task_list'])

    p['test_on_dataset'] = p['dataset_name']  # Taskonomy

    p['num_workers'] = 2 # just to cut down the cpu usage

    if 'Experiment_name_meta_test' in p.keys():
        p['Experiment_name'] = p['Experiment_name_meta_test']
    
    print(colored(p['setup'], 'blue'))

    if p['dataset_name'] == p['test_on_dataset']:

        fname = '/proj/ltu_mtl/users/x_ricup/results/Test/'+ p['dataset_name']+'/' +  'Test_' + p['Experiment_name']

    else:
        fname = '/proj/ltu_mtl/users/x_ricup/results/Test/cross/'+ 'Test_on_' +p['test_on_dataset']+'_'+ p['Experiment_name']

    if not os.path.exists(fname):
        os.makedirs(fname)
    else:
        print('folder already exist')

    writer = SummaryWriter(fname)

    p['dataset_name'] = p['test_on_dataset']
        
    print(colored('Retrieve dataset', 'blue'))
    test_dataset = get_test_dataset(p)
    print('Test samples %d -' %(len(test_dataset)))

    # if p['dataset_name'] == 'NYU':
    #     p['test_batch_size'] = 50

    test_loader = get_test_dataloader(p, test_dataset)

    
    batch_metric_dict =  collections.defaultdict(lambda: collections.defaultdict(list))
    # epochwise_val_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
   
    # model.eval()

    epoch_time = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            if (i == 20) and ('edge_texture' in p['task_list']):
                break
            # print(colored('val batch %d' %(i), 'yellow'))
            metric_dict = {}
            batch_time_start = time.time()
            # Forward pass
            images = batch['image']
            targets = batch['targets']
                
            # image,targets = batch
            images = images.cuda()        
            
            output, targets, best_epochs = test_model_single_MTL(p, images, targets)

            batch_time_stop = time.time()
            epoch_time += (batch_time_stop - batch_time_start)

            metric_dict = evaluation_metrics(p, output, targets)
              
            # update metric values of every batch to a dictionary
            for task,value in metric_dict.items():
                for k,v in value.items():
                    batch_metric_dict[task][k].append(v)
            
            # take a random image from the batch and predict the masks 
            # rand_img = np.random.randint(low = 0, high = images.shape[0])

            # rand_img = 20
            # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            # img = unorm(images[rand_img,:,:,:])
            # img1 = images[rand_img,:,:,:].permute(1,2,0)
            # img1 = img1.cpu()
            # writer.add_image('ground_truth', img,i)
            # # img_to_plot = img.permute(1,2,0).cpu().numpy()
            # img_to_plot = transforms.ToPILImage()(img).convert("RGB")
            

            # for task in p['task_list']:
            #     # if (task == 'class_scene') or (task == 'class_object'):
            #     #     gt, pred = find_top_3_classes(p, task, output[task][rand_img,:], targets[task][rand_img,:]) 
            #     #     gt_img = text_on_image(gt, img1)                    
            #     #     writer.add_image(task+ '_ground truth', gt_img,i)
            #     #     pred_img = text_on_image(pred, img1)
            #     #     writer.add_image(task + '_prediction', pred_img,i)

            #     #     if i % 10 == 0:
            #     #         pred_img = pred_img.permute(1,2,0).cpu().numpy()
            #     #         plt.imsave(fname+'/'+task+'_output_'+str(i)+ '.png', pred_img)




            #     if 'segmentsemantic' in p['task_list']:                    

            #         target = targets['segmentsemantic'][rand_img,0,:,:]
            #         # target = torch.squeeze(target,0)
            #         out = F.interpolate(output['segmentsemantic'], size=(256,256), mode="bilinear")
            #         out = F.softmax(out, dim = 1)
            #         out = out[rand_img,:,:,:]
            #         out = torch.argmax(out, dim = 0)

            #         if p['mode'] == 'binary':
            #             out = draw_binary_segmentation_map(p,out).permute(2,0,1)
            #             writer.add_image('segmentation_output', out, epoch)              
            #         else:
            #             if p['dataset_name'] == 'Taskonomy':
            #                 out = draw_segmentation_map_taskonomy(out).permute(2,0,1)
            #                 tar = draw_segmentation_map_taskonomy(target).permute(2,0,1)
            #             elif p['dataset_name'] == 'NYU':
            #                 out = draw_segmentation_map_NYU(out).permute(2,0,1)
            #                 tar = draw_segmentation_map_NYU(target).permute(2,0,1)
            #             else:
            #                 print('dataset not found')
            #             writer.add_image('Expected_output', tar, i)
            #             writer.add_image('segmentation_output', out, i)                        
                        
            #         # out_to_plot = out.permute(1,2,0).cpu().numpy()
            #         # overlay_img = cv2.addWeighted(img_to_plot, 1.0, out_to_plot, 0.4,0.0)
            #         # cv2.imwrite(fname+'seg_output_'+str(i)+'.png', overlay_img)
            #         out_to_plot = transforms.ToPILImage()(out).convert("RGB")
            #         overlay_img = Image.blend(img_to_plot, out_to_plot, 0.4)
            #         overlay_img.save(fname+'/overlay_seg_output_'+str(i)+'.png')
            #         # overlay_img.show()
            #         tar = tar.permute(1,2,0)
            #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            #         ax1.imshow(img1)
            #         ax1.set_title('Input image')
            #         ax2.imshow(tar)
            #         ax2.set_title('GT')
            #         ax3.imshow(out.permute(1,2,0))
            #         ax3.set_title('Pred.')
            #         plt.savefig(fname+'/seg_output_'+str(i)+ '.png', dpi=400)



            #     if 'depth_euclidean' in p['task_list']:
            #         tar = F.interpolate(targets['depth_euclidean'], size=(256,256), mode="bilinear")
            #         tar = tar[rand_img,:,:,:]
            #         writer.add_image('depth_ground_truth', tar, i)  

            #         out = F.interpolate(output['depth_euclidean'], size=(256,256), mode="bilinear")
            #         out = out[rand_img,:,:,:]
            #         writer.add_image('depth_predicted_output', out, i)

                    
            #         # out = torch.stack((out,out,out), dim = 0) 
            #         # out=out.squeeze(1)
            #         # print(out.shape)
                    
            #         out = out.permute(1,2,0).cpu().numpy()
            #         # img = sns.heatmap(out[:,:,0],cmap = 'YlGnBu', cbar = False) # 
            #         # plt.imsave(fname+'/depth_euclidean_output_'+str(i)+ '.png', img)
            #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            #         ax1.imshow(img1)
            #         ax1.set_title('Input image')
            #         ax2.imshow(tar.permute(1,2,0).cpu().numpy())
            #         ax2.set_title('GT')
            #         ax3.imshow(out)
            #         ax3.set_title('Pred.')
            #         plt.savefig(fname+'/depth_euclidean_output_'+str(i)+ '.png', dpi=400)




            #     if  'edge_texture' in p['task_list']:
            #         tar = F.interpolate(targets['edge_texture'], size=(256,256), mode="bilinear")
            #         tar = tar[rand_img,:,:,:]
            #         writer.add_image('edge_texture_truth', tar, i)
            #         # out = torch.sigmoid(output['edge_texture'])
            #         out = F.interpolate(output['edge_texture'], size=(256,256), mode="bilinear")
            #         out = out[rand_img,:,:,:]
            #         writer.add_image('edge_texture_output', out, i)

            #         # if i % 10 == 0:
            #         img_to_plt = out.permute(1,2,0).cpu().numpy()                        
            #         # plt.imsave(fname+'/edge_texture_output_'+str(i)+ '.png', img_to_plt)
            #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            #         ax1.imshow(img1)
            #         ax1.set_title('Input image')
            #         ax2.imshow(tar.permute(1,2,0).cpu().numpy())
            #         ax2.set_title('GT')
            #         ax3.imshow(img_to_plt)
            #         ax3.set_title('Pred.')
            #         plt.savefig(fname+'/edge_texture_output_'+str(i)+ '.png', dpi=400)


            #     if 'surface_normal' in p['task_list']:
            #         tar = F.interpolate(targets['surface_normal'], size=(256,256), mode="bilinear")
            #         tar = tar[rand_img,:,:,:]
            #         writer.add_image('surface_normal_truth', tar, i)                    
            #         out = F.interpolate(output['surface_normal'], size=(256,256), mode="bilinear")
            #         out = out[rand_img,:,:,:]
            #         writer.add_image('surface_normal_output', out, i)

            #         # if i % 10 == 0:
            #         img_to_plt = out.permute(1,2,0).cpu().numpy()                        
            #         # plt.imsave(fname+'/surface_normal_output_'+str(i)+ '.png', img_to_plt)
            #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            #         ax1.imshow(img1)
            #         ax1.set_title('Input image')
            #         ax2.imshow(tar.permute(1,2,0).cpu().numpy())
            #         ax2.set_title('GT')
            #         ax3.imshow(img_to_plt)
            #         ax3.set_title('Pred.')
            #         plt.savefig(fname+'/surface_normal_output_'+str(i)+ '.png', dpi=400)






    metric = {task: {m: np.mean(val) for m, val in values.items()} for task, values in batch_metric_dict.items()}
    print('test_metrics :', metric)
    print('---------------------------------')

    test_metrics = {}
    test_metrics['batchwise_metric'] = batch_metric_dict
    test_metrics['avg_metrics'] = metric

    f = open(fname + '/'+ "test_metric_dict.pkl","wb")
    pickle.dump(test_metrics,f)
    f.close()

    # best_epoch = checkpoint['epoch']

    print('best_epoch: ', best_epochs)
    print('---------------------------------')


    ##### printing train and validation metrics for record keeping 
    if os.path.exists(dir + 'training_dict.pkl'):
        with open(dir + 'training_dict.pkl', 'rb') as f:
            data = pickle.load(f)
        
        train_metrics = {}
        
        for task in p['task_list']:
            best_epoch = best_epochs[task]-1
            train_metrics[task] = {}
            metric_keys = data['metric'][task].keys()
            for metric in metric_keys:
                train_metrics[task][metric] = data['metric'][task][metric][best_epoch]

        print('train metrics:',train_metrics )

    print('---------------------------------')
    if os.path.exists(dir + 'validation_dict.pkl'):
        with open(dir + 'validation_dict.pkl', 'rb') as f:
            data = pickle.load(f)
        
        val_metrics = {}
        
        for task in p['task_list']:
            best_epoch = best_epochs[task]-1
            val_metrics[task] = {}
            metric_keys = data['metric'][task].keys()
            for metric in metric_keys:
                val_metrics[task][metric] = data['metric'][task][metric][best_epoch]

        print('val metrics:',val_metrics )





if __name__ == "__main__":
    main()



##
