import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.io import read_image
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
# from convert16_to_8_bit import map_uint16_to_uint8
import cv2


def normalize_16bit(resized):
    pixels = np.asarray(resized)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate global mean and standard deviation
    mean, std = pixels.mean(), pixels.std()
    #print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = np.clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    # confirm it had the desired effect
    mean, std = pixels.mean(), pixels.std()
    return pixels,mean,std


class NYUDataset(Dataset):
    def __init__(self, p, data_dir, set, transform):
        self.data_dir = data_dir   
        self.splits_dir = os.path.join(self.data_dir, 'gt_sets') 
        self.set = set
        with open(self.splits_dir+'/'+ self.set+'.txt', 'r') as f:
            self.lines = f.read().splitlines()
        self.transform = transform
        # self.images = self.data_dict['X'+ self.set]
        # self.seg_labels = self.data_dict['Y'+ self.set]
        self.task_list = p['task_list']
        if ('percent_data_for_training' in p.keys()) and (self.set != 'test'):
            self.percent_data = p['percent_data_for_training']
        else:
            self.percent_data = 100
        


    def __len__(self):
        len_data = int((self.percent_data/100)* len(self.lines))
        return len_data
        
        # return len(self.lines)        


    def __getitem__(self, idx): 
        sample= {}            
        image_name = self.data_dir +'/'+ 'images' +'/'+ self.lines[idx] +'.jpg'
        sample['image'] = Image.open(image_name).convert('RGB')
        

        sample['targets']={}
        for task in self.task_list:
            if task == 'segmentsemantic':
                label_name = self.data_dir +'/'+ 'segmentation' +'/'+ self.lines[idx] +'.png'
                img = read_image(label_name)
                semseg = torch.cat((img,img,img), dim = 0)
                sample['targets']['segmentsemantic'] = semseg


            elif task == 'depth_euclidean':
                label_name = self.data_dir +'/'+ 'depth' +'/'+ self.lines[idx] +'.npy'
                depth_img = np.load(label_name, allow_pickle= True)
                depth_img = depth_img/depth_img.max()
                depth_img = cv2.resize(depth_img, (256,256), interpolation = cv2.INTER_CUBIC) 
                depth_img = torch.tensor(depth_img) 
                depth_img = depth_img.unsqueeze(0)  
                # depth_img = torch.cat((depth_img,depth_img,depth_img), dim = 0)               
                sample['targets']['depth_euclidean'] = depth_img

            elif task == 'surface_normal':
                label_name = self.data_dir +'/'+ 'normals' +'/'+ self.lines[idx] +'.npy'
                sn_img = np.load(label_name, allow_pickle= True)
                sn_img = (sn_img - sn_img.min())/(sn_img.max() - sn_img.min())
                sn_img = cv2.resize(sn_img, (256,256), interpolation=cv2.INTER_NEAREST) 
                sn_img = np.transpose(sn_img, (2,0,1))
                
                sample['targets']['surface_normal'] = torch.tensor(sn_img)

            elif task == 'edge_texture':
                label_name = self.data_dir +'/'+ 'edge' +'/'+ self.lines[idx] +'.npy'
                edge_img = np.load(label_name, allow_pickle= True)
                edge_img = cv2.resize(edge_img, (256,256), interpolation=cv2.INTER_NEAREST)
                edge_img = torch.tensor(edge_img)                 
                edge_img = edge_img.unsqueeze(0)
                edge_img = torch.cat((edge_img,edge_img,edge_img), dim = 0)  

                sample['targets']['edge_texture'] = edge_img/edge_img.max()

            else:
                print('Task not found :', task)

        if self.transform:            
            sample = self.transform(sample)

        return sample
    




class TaskonomyDataset(Dataset):
    def __init__(self,p, data_dir, data_split_ids, task_list, set, indices_for_class_scene, indices_for_class_object, fold, transform):        
        self.data_dir = data_dir
        self.data_split_ids = data_split_ids
        self.set = set.lower()   ## train, test, val
        self.fold = fold  ## fold = 0,1,2 ...
        self.task_list = task_list     # class_object, class_scene, segmentsemantic
        self.transform = transform        
        self.indices_for_class_scene = indices_for_class_scene
        self.indices_for_class_object = indices_for_class_object  
        self.lower_bound, self.upper_bound = 0, 65535      
        self.lut = np.concatenate([np.zeros(self.lower_bound, dtype=np.uint16),  
                    np.linspace(0, 255, self.upper_bound - self.lower_bound).astype(np.uint16),
                    np.ones(2**16 - self.upper_bound, dtype=np.uint16) * 255])
        if self.fold is None:
            self.data_ids = self.data_split_ids[self.set]
        else:
            self.fname = 'train_fold'+str(self.fold)
            self.data_ids = self.data_split_ids[self.fname]

        self.prior_factor = np.load("/proj/ltu_mtl/dataset/taskonomy_dataset/utilities/prior_factor.npy")

        if ('percent_data_for_training' in p.keys()) and (self.set != 'test'):
            self.percent_data = p['percent_data_for_training']
        else:
            self.percent_data = 100


    def __len__(self):
        len_data = int((self.percent_data/100)* len(self.data_ids))
        return len_data
        # return len(self.data_ids)
        # return 10000


        


    def __getitem__(self, idx):        
        sample= {}
        ### read RGB input 
        img_path = self.data_dir + 'rgb/taskonomy/' + self.data_ids[idx] + 'rgb.png'
        # print(img_path)
        image = Image.open(img_path).convert("RGB")

        sample['image'] = image
        
        sample['targets'] = {}
        # if self.transform:
        #     image = self.transform(image)

        
        list = self.task_list
        for task in list:
            if task == 'class_scene':
                label_path =  self.data_dir + task + '/taskonomy/' + self.data_ids[idx] + 'class_places.npy'
                if os.path.exists(label_path):
                    labels_365 = np.load(label_path)
                    l_scene = np.take(labels_365,self.indices_for_class_scene)  
                    sample['targets']['class_scene'] = (l_scene - np.min(l_scene))/(l_scene-np.min(l_scene)).sum()  ## normalizing the labels such that sum = 1   
                    # sample['targets']['class_scene'] = l_scene
                             
                else:
                    break

            elif task == 'class_object':
                label_path =  self.data_dir + task + '/taskonomy/' + self.data_ids[idx] + task + '.npy'
                if os.path.exists(label_path):
                    labels_1000 = np.load(label_path)
                    l = np.take(labels_1000,self.indices_for_class_object)                     
                    sample['targets']['class_object'] = (l - np.min(l))/(l-np.min(l)).sum()  ## normalizing the labels such that sum = 1
                    # sample['targets']['class_object'] = l
                    
                else:
                    break
                
            elif task == 'segmentsemantic':
                label_path =  self.data_dir + task + '/taskonomy/' + self.data_ids[idx] + task +'.png'
                if os.path.exists(label_path):
                    img = read_image(label_path)
                    img[img == 0] = 1
                    img = img - 1
                    img = torch.cat((img,img,img), dim = 0)                   

                    sample['targets']['segmentsemantic'] = img.float()  

                else:
                    break

            elif task == 'depth_euclidean':
                label_path =  self.data_dir + task + '/taskonomy/' + self.data_ids[idx] + task +'.png'
                if os.path.exists(label_path):
                    # # file = np.asarray(Image.open(label_path))   # read 16 bit image as PIL
                    file = cv2.imread(label_path)  
                    if file is None:
                        print('file is none')
                        break
                    else: 
                        pixels = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)               
                        pixels = cv2.resize(pixels, (256,256), interpolation = cv2.INTER_CUBIC) 
                        # pixels = pixels/pixels.max() 
                        pixels, mean, std = normalize_16bit(pixels) 
                        pixels = torch.tensor(pixels) 
                        # pixels = pixels.permute(2,0,1) 
                        
                        
                        pixels = pixels[:,:,0]     
                        pixels = torch.unsqueeze(pixels, 0)                                   
                    
                        sample['targets']['depth_euclidean'] = pixels
                else:
                    break
            
            elif task == 'surface_normal':
                label_path =  self.data_dir + task + '/taskonomy/' + self.data_ids[idx] + 'normal' +'.png'

                if os.path.exists(label_path):  
                    img = cv2.imread(label_path)
                    if img is None:
                        print('file is none')
                        break
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                        img = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)
                        img = img/img.max()
                        img = torch.tensor(img)
                        img = img.permute(2,0,1) 
                        sample['targets']['surface_normal'] = img

                else:
                    # print('image does not exist', label_path)
                    break

            elif task == 'edge_texture':
                label_path =  self.data_dir + task + '/taskonomy/' + self.data_ids[idx] + 'edge_texture' +'.png'
                
                if os.path.exists(label_path):  
                    img = cv2.imread(label_path)
                    if img is None:                        
                        print('file is none')
                        break
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                        img = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)                      
                        img = img/img.max()
                        # img = np.transpose(img, (2,0,1))
                        img = torch.tensor(img)
                        img = img.permute(2,0,1)
                        sample['targets']['edge_texture'] = img

                else:
                    # print('image does not exist', label_path)
                    break

            else:
                print('Task not found :', task)
                # labels[task] = []
            

        if self.transform:
            # sample['image'] = self.transform(sample['image'])
            sample = self.transform(sample)


        if len(sample['targets']) == len(self.task_list):
            return sample
        else:
            return None