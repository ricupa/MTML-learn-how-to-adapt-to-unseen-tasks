
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
from torchvision import datasets, models, transforms

from utils.utils_common import *
from train.train_utils import train_function
from validation.val_utils import val_function
import collections
import time
import yaml
import dill as pickle
import warnings
warnings.filterwarnings('ignore')
import cv2
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import PIL



def draw_seg_map_NYU(outputs):
    
    label_map = [               
            #    (0,0,0), # background
               (255,255,255),  # wall
               (0, 255, 0),
               (0, 0, 255),
               (128, 0, 0), # 
               (0, 128, 0), # 
               (128, 128, 0), # 
               (0, 0, 128), # 
               (128, 0, 128), # 
               (0, 128, 128), # 
               (128, 128, 128), # 
               (64, 0, 0), # 
               (192, 0, 0), # 
               (64, 128, 0), # 
               (192, 128, 0), # 
               (64, 0, 128), # 
               (192, 0, 128), # 
               (64, 128, 128), # 
               (192, 128, 128), #
               (0, 64, 0), # 
               (0, 0, 64),
               (0, 192, 0),
               (0, 0, 192),
               (32,0,0),
               (0, 32, 0),
               (0, 0, 32),
               (255,128,128),
               (64,64, 128),
               (32, 255, 0),
               (255, 32, 0),
               (0, 32, 255),
               (192, 0,32),
               (32, 0, 192),
               (64, 32, 128),
               (128, 128, 32),
               (128, 64, 32),
               (192, 32, 64),
               (0, 192, 64),
               (192, 0, 255),
               (255, 64, 192),
               (0,0,0)               
            ]
    labels = outputs    
    # labels = torch.argmax(outputs, dim=0).detach().cpu().numpy() 
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=1)
    return torch.tensor(segmented_image)


name = "/proj/ltu_mtl/users/x_ricup/results/NYU/6_edge_texture_trial_3/"

config_exp = name+ 'config_file.yaml'
p = create_config(config_exp)

out_folder = '/proj/ltu_mtl/users/x_ricup/results/images/'+p['dataset_name']  +'/'

idx = '283'
set = 'train_'   ### val, test
imgname = '/proj/ltu_mtl/dataset/NYU_dataset/NYUD_MT/images/'+set+ idx+'.jpg'


##### for NYU dataset

# img = PIL.Image.open(imgname)
# img1 = img.resize((256,256), PIL.Image.ANTIALIAS)
# transform_v = [transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
# img_transform = transforms.Compose(transform_v) 
# img2 = img_transform(img)
# img2 = img2.unsqueeze(0)

# img1.save(out_folder+'image.jpg')



## groundtruths 

# segimgname = '/home/ricupa/Documents/M3TL/NYU_dataset/NYUD_MT/segmentation/'+set+ idx+'.png'
# img = cv2.imread(segimgname)[:,:,0]
# # img = np.transpose(img, (2,0,1))
# seg = draw_seg_map_NYU(img)
# segimg = np.asarray(seg)
# plt.imshow(segimg)
# cv2.imwrite(out_folder+'segmask.jpg', segimg)



# depthimgname = '/home/ricupa/Documents/M3TL/NYU_dataset/NYUD_MT/depth/'+set+ idx+'.npy'
# img = np.load(depthimgname)
# depth_gt = cv2.resize(img, (256,256))
# plt.imshow(depth_gt, cmap = 'gray')
# plt.axis('off')
# plt.savefig(out_folder+'depthmask.jpg', dpi=400)


# normalimgname = '/home/ricupa/Documents/M3TL/NYU_dataset/NYUD_MT/normals/'+set+ idx+'.npy'
# img = np.load(normalimgname)
# img = (img - img.min())/(img.max()-img.min())
# img = img[50:,50:-50,:]
# normal_gt = cv2.resize(img, (256,256))
# plt.imshow(normal_gt)
# plt.axis('off')
# plt.savefig(out_folder+'surface_normal_mask.jpg', dpi=400)

# edgeimgname = '/home/ricupa/Documents/M3TL/NYU_dataset/NYUD_MT/edge/'+set+ idx+'.npy'
# img = np.load(edgeimgname)
# edge_gt = cv2.resize(img, (256,256))
# plt.imshow(edge_gt, cmap = 'gray')
# plt.axis('off')
# plt.savefig(out_folder+'edge_mask.jpg', dpi=400)



#### load model 
#### inference 
# pred = {}
# img2 = img2.cuda()
# # print(img2.shape)

# for task in p['task_list']:
#     task_chkpt = name + task+'_checkpoint.pt'     

#     if os.path.exists(task_chkpt): 
#             print('task checkpoint exist')   
#             checkpoint = torch.load(task_chkpt)
#             model = checkpoint['model']
#     else:
#         print('task checkpoint not exist') 
#         checkpoint = torch.load(name + 'checkpoint.pt')
#         model = checkpoint['model']   
    
#     model.eval()
    
#     # print(img2.shape)
#     temp= model(img2)

#     pred[task] = temp[task]




# if 'segmentsemantic' in p['task_list']:
#     out = pred['segmentsemantic']
#     out = F.interpolate(out, size=(256,256), mode="bilinear")
#     out = F.softmax(out, dim = 1)
#     out = out[0,:,:,:]
#     out = torch.argmax(out, dim = 0)   
#     out = draw_segmentation_map_NYU(out)
#     out = np.asarray(out)
#     # cv2.imwrite(out_folder+name.split('/')[-2]+'_seg_output.jpg', out)
#     plt.imshow(out)
#     plt.axis('off')
#     plt.savefig(out_folder+name.split('/')[-2]+'_seg_output.jpg', dpi=400)
    

# if 'depth_euclidean' in p['task_list']:
#     out = F.interpolate(pred['depth_euclidean'], size=(256,256), mode="bilinear")
#     out = out[0,0,:,:]
#     out = F.sigmoid(out)
#     out = out.detach().cpu().numpy()
#     plt.imshow(out, cmap = 'gray')
#     plt.axis('off')
#     plt.savefig(out_folder+name.split('/')[-2]+'_depth_output.jpg', dpi=400)

# er+n
# if 'surface_normal' in p['task_list']:
#     out = F.interpolate(pred['surface_normal'], size=(256,256), mode="bilinear")
#     out = out[0,:,:,:]
#     out = F.sigmoid(out)
#     out = out.permute(1,2,0).detach().cpu().numpy()     
#     # out = (img - out.min())/(out.max()-out.min())
#     # out = cv2.blur(out,(5,5)) 
#     # out = cv2.blur(normal_gt,(11,11)) 
#     plt.imshow(out)
#     plt.axis('off')
#     plt.savefig(out_foldame.split('/')[-2]+'_surface_output.jpg', dpi=400)
    

# if 'edge_texture' in p['task_list']:
#     out = F.interpolate(pred['edge_texture'], size=(256,256), mode="bilinear")
#     out = out[0,:,:,:]
#     out = F.sigmoid(out)
#     out = out.permute(1,2,0).detach().cpu().numpy()
#     # out = out.detach().cpu().numpy()
    
#     # out = (img - out.min())/(out.max()-out.min())
#     # out = cv2.blur(out,(13,13)) 
#     # out = cv2.blur(edge_gt,(4,4)) 
#     plt.imshow(out)
#     plt.axis('off')
    # plt.savefig(out_folder+name.split('/')[-2]+'_edge_output.jpg', dpi=400)


    # kernel = np.ones((5,5), np.uint8)  
    # out = cv2.dilate(out, kernel, iterations=5) 
    # kernel = np.ones((2,2), np.uint8)  
    # out = cv2.erode(out, kernel, iterations=5)  




