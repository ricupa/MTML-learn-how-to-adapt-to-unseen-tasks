import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from all_model.backbone import ResNet_backbone,EfficientNetb7_backbone
from all_model.models_head import classification_head,segmentation_head_aspp
# from utils.metrics import *
from data.dataset import TaskonomyDataset,NYUDataset
import pickle
from torch.utils.data import Dataset, DataLoader
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import copy
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score
import random

def get_backbone(p):
    """ Return the backbone """

    if p['backbone_model'] == 'resnet18':
        from all_model.resnet import resnet18
        backbone = resnet18(p['pretrained'])        
        backbone_channels = 512
    
    elif p['backbone_model'] == 'resnet50':
        from all_model.resnet import resnet50
        backbone = resnet50(p['pretrained'])
        backbone_channels = 2048
        from all_model.res_dialated import ResnetDilated 
        backbone = ResnetDilated(backbone)

        
        for ct, child in enumerate(backbone.children()):
            for param in child.parameters():
                if (p['pretrained'] == True ) and (p['finetuning'] == True):
                    param.requires_grad = True
                elif (p['pretrained'] == True ) and (p['finetuning'] == False):
                    param.requires_grad = False
                elif p['pretrained'] == False:
                    param.requires_grad = True
    

    elif p['backbone_model'] == 'resnet101':   ### not using at present 
        from all_model.resnet import resnet101
        backbone = resnet101(p['pretrained'])
        backbone_channels = 2048
        from all_model.res_dialated import ResnetDilated 
        backbone = ResnetDilated(backbone)

        if p['pretrained'] == True:
            for ct, child in enumerate(backbone.children()):        
                if ct < 6:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            for ct, child in enumerate(backbone.children()):
                for param in child.parameters():
                        param.requires_grad = True


    elif p['backbone_model'] == 'efficientnetb7':
        import torchvision.models as models

        if p['pretrained'] == True:
            efficientnet_b7 = models.efficientnet_b7(pretrained=True)
            for ct, child in enumerate(efficientnet_b7.children()):
                if ct < 6:      
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            efficientnet_b7 = models.efficientnet_b7(pretrained=False)
            for param in efficientnet_b7.children():
                param.requires_grad = True 

        backbone = EfficientNetb7_backbone(efficientnet_b7)
        
        backbone_channels = 2048

    else:
        raise NotImplementedError

    return backbone, backbone_channels





def get_head(p, in_channels, task):
    if (task == 'class_scene'):
        num_classes = 63
        return classification_head(in_channels = in_channels, num_classes = num_classes)
    
    elif (task == 'class_object'):
        num_classes = 100
        return classification_head(in_channels = in_channels, num_classes = num_classes)

    elif task == 'segmentsemantic':   
        if p['mode'] == 'binary' :
            num_classes = 1
        else:
            if p['dataset_name'] =='Taskonomy':
                num_classes = 17
            elif p['dataset_name'] == 'NYU':
                num_classes = 41
        return segmentation_head_aspp(p=p, in_channels=in_channels, num_classes = num_classes)


    elif task == 'depth_euclidean':
        num_classes = 1
        return segmentation_head_aspp(p=p,in_channels=in_channels, num_classes = num_classes)
    
    elif task == 'surface_normal':
        num_classes = 3
        return segmentation_head_aspp(p=p,in_channels=in_channels, num_classes = num_classes)
    
    elif task == 'edge_texture':
        num_classes = 3
        return segmentation_head_aspp(p=p,in_channels=in_channels, num_classes = num_classes)

    else:
        print('Task not found')


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)
    
    if p['setup'] == 'single_task':
        from all_model.models import SingleTaskModel
        task = p['task_list'][0]
        head = get_head(p, backbone_channels, task)
        model = SingleTaskModel(backbone, head, task)


    elif p['setup'] == 'multi_task':
        if p['model'] == 'baseline':
            from all_model.models import MultiTaskModel
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p['task_list']})

            for params in heads.parameters():
                params.requires_grad = True

            model = MultiTaskModel(backbone, heads, p['task_list'])

        
        else:
            raise NotImplementedError('Unknown model {}'.format(p['model']))


    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))
    

    return model



def get_loss(p, task):
    """ Return loss function for a specific task """

    if task == 'class_object':  
        from utils.loss_functions import softmax_cross_entropy_with_softtarget
        criterion = softmax_cross_entropy_with_softtarget() 
        

    elif task == 'class_scene':
        from utils.loss_functions import softmax_cross_entropy_with_softtarget
        criterion = softmax_cross_entropy_with_softtarget()
    
    elif task == 'segmentsemantic':
        # from utils.segmentation_loss import MultiTverskyLoss
        if p['dataset_name'] == 'NYU':
            alpha = 0.3
            beta = 0.7
            gamma = 2
        elif p['dataset_name'] == 'Taskonomy':
            alpha = 0.3
            beta = 0.7
            gamma = 2.5
        else:
            print('Unknown dataset')
        # criterion = MultiTverskyLoss(p =p, alpha=alpha, beta=beta, gamma=gamma)  

        from utils.segmentation_loss import Seg_cross_entropy_loss
        criterion = Seg_cross_entropy_loss(p)

     

    elif task == 'depth_euclidean':
        from utils.loss_functions import Depth_combined_loss, DepthLoss, RMSE_log
        criterion = Depth_combined_loss()
        # criterion = DepthLoss()
        # criterion = RMSE_log()
    
    elif task == 'surface_normal':
        from utils.loss_functions import surface_normal_loss
        criterion = surface_normal_loss()
    
    elif task == 'edge_texture':
        from utils.loss_functions import edge_loss
        criterion =  edge_loss()

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                    'class_object, class_scene, segmentsemantic')

    return criterion



def get_criterion(p):
    """ Return training criterion for a given setup """
    if p['setup'] == 'single_task':
        task = p['task_list'][0]
        loss_ft = get_loss(p, task)
        return loss_ft

    
    elif p['setup'] == 'multi_task':
        if p['model'] == 'baseline': # Fixed weights            
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p['task_list']})                
            return loss_ft        
        else:
            raise NotImplementedError('Unknown loss scheme {}'.format(p['loss_kwargs']['loss_scheme']))
    
    elif p['setup'] == 'meta_multi_test':
        task_list = p['meta_test_task_list'] + p['task_list']
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in task_list})

    
    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))




def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """
    params = model.parameters()
    if p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,lr=p['optimizer_params']['learning_rate'], betas=p['optimizer_params']['betas'], weight_decay=p['optimizer_params']['weight_decay'])

    elif p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params,lr=p['optimizer_params']['learning_rate'], momentum=0.9, weight_decay=p['optimizer_params']['weight_decay'])

    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=p['optimizer_params']['learning_rate'], betas=p['optimizer_params']['betas'], weight_decay=p['optimizer_params']['weight_decay'], eps=1e-07, amsgrad= True)

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))   
    
    return optimizer


def get_meta_optimizer(p, model):
    """ Return optimizer for a given model and setup """
    params = model.parameters()
    if p['meta']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,lr=p['meta']['learning_rate'], betas=p['meta']['betas'], weight_decay=p['optimizer_params']['weight_decay'])

    elif p['meta']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params,lr=p['meta']['learning_rate'], momentum=0.9, weight_decay=p['meta']['weight_decay'])

    elif p['meta']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=p['meta']['learning_rate'], betas=p['meta']['betas'], weight_decay=p['meta']['weight_decay'], eps=1e-07, amsgrad= True)

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))   
    
    return optimizer



def get_scheduler(p, optimizer):
    """ Adjust the learning rate """

    if p['scheduler'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=p['scheduler_params']['lr_decay_factor'], patience=p['scheduler_params']['patience'],verbose = True)
        
        
    else:
        raise ValueError('Invalid scheduler {}'.format(p['scheduler']))

    return scheduler



def get_transformations(p):
    ### return transformations 
    from utils import utils_transforms as tr
    #tr.RandomRotate(30),
    transform_tr = [tr.RandomHorizontalFlip(),
                    tr.FixedResize((256,256)), tr.ToTensor(),
                    tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    transform_v = [tr.FixedResize((256,256)), tr.ToTensor(), 
                    tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]


    train_transform = transforms.Compose(transform_tr)
    val_transform = transforms.Compose(transform_v)                       

    return train_transform, val_transform





def get_dataset(p,fold=None):

    if p['dataset_name'] == 'Taskonomy':

        p['data_dir'] = "../dataset/taskonomy_dataset/"
        p['data_split'] = "annotations/data_split_ids_nofolds.pkl"   
        p['class_scene_idx_name_file'] = 'annotations/scene_names_idx_label_63.pkl'
        p['class_object_idx_name_file'] = 'annotations/object_names_idx_label_100.pkl'


        file = open(p['data_split'], "rb")
        data_split_ids = pickle.load(file)  

        # for task = 'class_scene'
        file_to_read = open(p['class_scene_idx_name_file'], "rb")  # pickle file of the 63 useful index and class scene/places names 
        loaded_dictionary = pickle.load(file_to_read)
        indices_for_class_scene = np.asarray(list(loaded_dictionary.keys()))

        #### for task = 'class_object'
        file_to_read = open(p['class_object_idx_name_file'], "rb")  # pickle file of the 100 useful index and class object names  
        loaded_dictionary = pickle.load(file_to_read)
        indices_for_class_object = np.asarray(list(loaded_dictionary.keys()))

        train_transform, val_transform = get_transformations(p)

        train_dataset = TaskonomyDataset(p=p,data_dir = p['data_dir'], 
                                    data_split_ids = data_split_ids, 
                                    task_list = p['task_list'], 
                                    set = 'train', 
                                    indices_for_class_scene = indices_for_class_scene ,
                                    indices_for_class_object = indices_for_class_object, 
                                    fold=None, 
                                    transform= train_transform)

        val_dataset = TaskonomyDataset(p=p,data_dir = p['data_dir'], 
                                    data_split_ids = data_split_ids, 
                                    task_list = p['task_list'], 
                                    set = 'val', 
                                    indices_for_class_scene = indices_for_class_scene ,
                                    indices_for_class_object = indices_for_class_object, 
                                    fold=None, 
                                    transform= val_transform)                            
        

    elif p['dataset_name'] == 'NYU':

        p['data_dir'] = "../dataset/NYU/NYUD_MT"

        train_transform, val_transform = get_transformations(p)

        train_dataset = NYUDataset(p = p, data_dir = p['data_dir'], set = 'train', transform= train_transform)

        val_dataset = NYUDataset(p= p, data_dir = p['data_dir'], set = 'val', transform= val_transform)
    
    else:
        print('Dataset not found')
    return train_dataset,val_dataset



def get_meta_transformations(p):
    ### return transformations 
    from utils import utils_transforms as tr
    #tr.RandomRotate(30),
    transform_tr = [tr.RandomHorizontalFlip(),
                    tr.FixedResize((256,256)), tr.ToTensor(),
                    tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    transform_v = [tr.RandomHorizontalFlip(),
                    tr.FixedResize((256,256)), tr.ToTensor(),
                    tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    train_transform = transforms.Compose(transform_tr)
    val_transform = transforms.Compose(transform_v)                       

    return train_transform, val_transform


def get_meta_dataset(p,fold=None):
    
    if p['dataset_name'] == 'Taskonomy':

        p['data_dir'] = "../dataset/taskonomy_dataset/"
        p['data_split'] = "annotations/data_split_ids_nofolds.pkl"   
        p['class_scene_idx_name_file'] = 'annotations/scene_names_idx_label_63.pkl'
        p['class_object_idx_name_file'] = 'annotations/object_names_idx_label_100.pkl'


        file = open(p['data_split'], "rb")
        data_split_ids = pickle.load(file)  

        # for task = 'class_scene'
        file_to_read = open(p['class_scene_idx_name_file'], "rb")  # pickle file of the 63 useful index and class scene/places names 
        loaded_dictionary = pickle.load(file_to_read)
        indices_for_class_scene = np.asarray(list(loaded_dictionary.keys()))

        #### for task = 'class_object'
        file_to_read = open(p['class_object_idx_name_file'], "rb")  # pickle file of the 100 useful index and class object names  
        loaded_dictionary = pickle.load(file_to_read)
        indices_for_class_object = np.asarray(list(loaded_dictionary.keys()))

        train_transform, val_transform = get_meta_transformations(p)

        train_dataset = TaskonomyDataset(p=p,data_dir = p['data_dir'], 
                                    data_split_ids = data_split_ids, 
                                    task_list = p['task_list'], 
                                    set = 'train', 
                                    indices_for_class_scene = indices_for_class_scene ,
                                    indices_for_class_object = indices_for_class_object, 
                                    fold=None, 
                                    transform= train_transform)

        val_dataset = TaskonomyDataset(p=p,data_dir = p['data_dir'], 
                                    data_split_ids = data_split_ids, 
                                    task_list = p['task_list'], 
                                    set = 'val', 
                                    indices_for_class_scene = indices_for_class_scene ,
                                    indices_for_class_object = indices_for_class_object, 
                                    fold=None, 
                                    transform= val_transform)                            
        

    elif p['dataset_name'] == 'NYU':

        p['data_dir'] = "../dataset/NYU/NYUD_MT"

        train_transform, val_transform = get_transformations(p)

        train_dataset = NYUDataset(p = p, data_dir = p['data_dir'], set = 'train', transform= train_transform)

        val_dataset = NYUDataset(p= p, data_dir = p['data_dir'], set = 'val', transform= val_transform)
    
    else:
        print('Dataset not found')
    return train_dataset,val_dataset




def get_test_dataset(p):

    if p['dataset_name'] == 'Taskonomy':

        p['data_dir'] = "../dataset/taskonomy_dataset/"
        p['data_split'] = "annotations/data_split_ids_nofolds.pkl"   
        p['class_scene_idx_name_file'] = 'annotations/scene_names_idx_label_63.pkl'
        p['class_object_idx_name_file'] = 'annotations/object_names_idx_label_100.pkl'


        file = open(p['data_split'], "rb")
        data_split_ids = pickle.load(file)  

        # for task = 'class_scene'
        file_to_read = open(p['class_scene_idx_name_file'], "rb")  # pickle file of the 63 useful index and class scene/places names 
        loaded_dictionary = pickle.load(file_to_read)
        indices_for_class_scene = np.asarray(list(loaded_dictionary.keys()))

        #### for task = 'class_object'
        file_to_read = open(p['class_object_idx_name_file'], "rb")  # pickle file of the 100 useful index and class object names  
        loaded_dictionary = pickle.load(file_to_read)
        indices_for_class_object = np.asarray(list(loaded_dictionary.keys()))

        _ , test_transform = get_transformations(p)

        test_dataset = TaskonomyDataset(p=p,data_dir = p['data_dir'], 
                                    data_split_ids = data_split_ids, 
                                    task_list = p['task_list'], 
                                    set = 'test', 
                                    indices_for_class_scene = indices_for_class_scene ,
                                    indices_for_class_object = indices_for_class_object, 
                                    fold=None, 
                                    transform = test_transform)
    

    elif p['dataset_name'] == 'NYU':
        p['data_dir'] = "../dataset/NYU/NYUD_MT"

        _ , test_transform = get_transformations(p)        

        test_dataset = NYUDataset(p= p, data_dir = p['data_dir'], set = 'test', transform = test_transform)  
    
    else:
        print('Dataset not found')

    return test_dataset




def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))    
    if len(batch)  == 0:
        print('len of batch is 0') 
    return torch.utils.data.dataloader.default_collate(batch)


def seed_worker(worker_id):
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(p,train_dataset,val_dataset):

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(dataset = train_dataset, batch_size = p['train_batch_size'], collate_fn=collate_fn, shuffle = True, num_workers = p['num_workers'], drop_last=True)

    val_loader = DataLoader(dataset = val_dataset, batch_size = p['val_batch_size'], collate_fn=collate_fn, shuffle = True, num_workers = p['num_workers'], drop_last=True,worker_init_fn=seed_worker, generator=g)
    
    ### worker_init_fn=seed_worker, generator=g,
    return train_loader, val_loader


def get_meta_dataloader(p,train_dataset,val_dataset):
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = p['train_batch_size'], collate_fn=collate_fn, shuffle = True, num_workers = p['num_workers'], drop_last=True)

    val_loader = DataLoader(dataset = val_dataset, batch_size = p['val_batch_size'], collate_fn=collate_fn, shuffle = True, num_workers = p['num_workers'],drop_last=True)
    return train_loader, val_loader


def get_test_dataloader(p,test_dataset):

    g = torch.Generator()
    g.manual_seed(0)
    test_loader = DataLoader(dataset = test_dataset, batch_size = p['test_batch_size'], collate_fn=collate_fn, shuffle = True, drop_last=True, worker_init_fn=seed_worker, generator=g)

    return test_loader



def compute_valid_depth_mask(d1, d2=None):
    """Computes the mask of valid values for one or two depth maps    
    Returns a valid mask that only selects values that are valid depth value 
    in both depth maps (if d2 is given).
    Valid depth values are >0 and finite.
    """

    if d2 is None:
        valid_mask = torch.isfinite(d1)
        valid_mask[valid_mask] = (d1[valid_mask] > 0)
    else:
        valid_mask = torch.isfinite(d1) & torch.isfinite(d2)
        _valid_mask = valid_mask.clone()
        valid_mask[_valid_mask] = (d1[_valid_mask] > 0) & (d2[_valid_mask] > 0)
    return valid_mask



def get_valid_depth_values(pred,gt):
    valid_mask = torch.tensor(compute_valid_depth_mask(pred, gt),dtype=torch.uint8)
    depth_pred = torch.nan_to_num(pred*valid_mask, nan=0.0, posinf=1.0, neginf=0.0 )
    depth_gt = torch.nan_to_num(gt*valid_mask)
    return depth_pred, depth_gt


def one_hot(gt, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
    gt = gt.cuda()        
    input_shape = tuple(gt.shape)  # (N, H, W, ...)        
    new_shape = (input_shape[0], class_num) + input_shape[1:]        
    one_hot = torch.zeros(new_shape, device=torch.device('cuda'))
    # one_hot = torch.zeros(new_shape, dtype = torch.float)
    target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)  #
    # target = Variable(target, requires_grad = True)
    return target


class ModelLossWrapper(nn.Module):
    def __init__(self, p, loss_ft: nn.ModuleDict, model):
        super(ModelLossWrapper, self).__init__()
        self.task_num = len(p['task_list'])
        self.set_up = p['setup']                 
        self.loss_ft = loss_ft
        self.tasks = p['task_list']
        self.model = model
        self.task = p['task_list'][0] #for single task only          
        self.th = torch.tensor(p['threshold']).cuda()  # for binary segmentation  (redundant)
        self.seg_mode = p['mode']
        self.comb_loss = p['comb_loss']  # mode for balancing the loss
        p['flag'] = {}
        for task in p['task_list']:
            p['flag'][task] = 1
        self.flags = p['flag']
        self.dataset = p['dataset_name']

        if self.comb_loss == 'uncertainity':
            self.log_vars = nn.Parameter(torch.zeros(self.task_num))
        elif self.comb_loss == 'sum':
            self.weights = {}
            for task in self.tasks:
                self.weights[task] = torch.tensor(1.0)
        elif self.comb_loss == 'gradnorm':
            self.weights = nn.Parameter(torch.ones(self.task_num).float())
        else:
            print('Implementation error: task not found')




    def forward(self, images, targets, new_task_list):              

        pred  = self.model(images)     
   
        if 'depth_euclidean' in self.tasks:   
            new_dim = pred['depth_euclidean'].shape[-2:]
            targets['depth_euclidean'] = F.interpolate(targets['depth_euclidean'].type(torch.DoubleTensor), size= new_dim, mode="bilinear")      
            pred['depth_euclidean'] = torch.sigmoid(pred['depth_euclidean'])
            pred['depth_euclidean'], targets['depth_euclidean'] = get_valid_depth_values(pred['depth_euclidean'], targets['depth_euclidean'].cuda())
            # print(targets['depth_euclidean'].shape)
            # print(pred['depth_euclidean'].shape)
            assert targets['depth_euclidean'].shape == pred['depth_euclidean'].shape 

        if 'edge_texture' in self.tasks:
            pred['edge_texture'] = torch.sigmoid(pred['edge_texture'])
        
        if 'surface_normal' in self.tasks:
            pred['surface_normal'] = torch.sigmoid(pred['surface_normal'])


        if self.set_up == 'multi_task':            

            out= {}

            for task in self.tasks:
                if task in new_task_list:
                    # if list(self.flags.values()).sum() == len(self.tasks):
                    out[task] = self.loss_ft[task](pred[task], targets[task])
                else:
                    out[task] = torch.tensor(0 , dtype=torch.float64).cuda()
            
            # out = {task: self.loss_ft[task](pred[task], targets[task]) for task in self.tasks}

            if self.comb_loss == 'sum':

                loss = 0
                weighted_task_loss = {task: torch.mul(self.weights[task], out[task]) for task in self.tasks} 
                for task, val in weighted_task_loss.items():                    
                    loss += val * self.flags[task]

            elif self.comb_loss == 'uncertainity':
                # out_loss  = out
                weighted_task_loss = {}
                for i, task in enumerate(self.tasks):
                    pre = torch.exp(-self.log_vars[i])
                    weighted_task_loss[task] = pre * out[task] + self.log_vars[i]
                loss = 0
                for task,val in weighted_task_loss.items():
                    loss += val * self.flags[task]

    

            
            elif self.comb_loss == 'gradnorm':

                ##### from - https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py
                loss = 0
                weighted_task_loss = {task: torch.mul(self.weights[i], out[task]) for i, task in enumerate(self.tasks)} 
                for task,val in weighted_task_loss.items():
                    loss += val

            else:
                print('Implementation error : this mode of combining loss is not implemeted ')              
            
            out['total'] = loss

            return pred, targets, out, weighted_task_loss   ###self.log_vars.data.tolist()
            
        else:
            out = {self.task: self.loss_ft(pred[self.task], targets[self.task])}
            out['total'] = out[self.task]  
              
        
        return pred, targets, out, out  # the second return --out is of no use
            


def test_model_single_MTL(p, images, targets):  
    pred = {} 

    targets = {task: val.cuda() for task, val in targets.items()} 

    best_epochs = {}    

    for task in p['task_list']:
        task_chkpt = p['train_dir'] + task+'_checkpoint.pt'        
        if os.path.exists(task_chkpt): 
            print('task checkpoint exist')   
            checkpoint = torch.load(task_chkpt)
            model = checkpoint['model']
        else:
            print('task checkpoint not exist') 
            checkpoint = torch.load(p['train_dir'] + 'checkpoint.pt')
            model = checkpoint['model']               

        best_epochs[task] = checkpoint['epoch']
        model = model.cuda()
        model.eval()
        prediction  = model(images) 
        pred[task] = prediction[task] 


    if 'depth_euclidean' in p['task_list']:   
        new_dim = pred['depth_euclidean'].shape[-2:]
        targets['depth_euclidean'] = F.interpolate(targets['depth_euclidean'], size= new_dim, mode="bilinear")      
        pred['depth_euclidean'] = torch.sigmoid(pred['depth_euclidean'])
        pred['depth_euclidean'], targets['depth_euclidean'] = get_valid_depth_values(pred['depth_euclidean'], targets['depth_euclidean'])

        assert targets['depth_euclidean'].shape == pred['depth_euclidean'].shape 

    if 'edge_texture' in p['task_list']:
        pred['edge_texture'] = torch.sigmoid(pred['edge_texture'])

    
    if 'surface_normal' in p['task_list']:
        pred['surface_normal'] = torch.sigmoid(pred['surface_normal']) 


    
    return pred, targets, best_epochs



def find_top_3_classes(p, task, output, label):
    assert(output.shape == label.shape)              

    # since label and output both are probabilities, 
    # convert then into a multi-label output by taking the top-5 prob = 1  
    #   
    idx_gt = torch.topk(label,3).indices
    idx_pred = torch.topk(output,3).indices

    if task == 'class_scene':
        file_to_read = open(p['class_scene_idx_name_file'], "rb")  # pickle file of the 63 useful index and class scene/places names 
        annotation_dict = pickle.load(file_to_read)    

    if task == 'class_object':
        file_to_read = open(p['class_object_idx_name_file'], "rb")  # pickle file of the 100 useful index and class object names  
        annotation_dict = pickle.load(file_to_read) 

    gt_names = []
    for idx in idx_gt:
        gt_names.append(list(annotation_dict.values())[idx].split(',')[0])
    pred_names = []
    for idx in idx_pred:
        pred_names.append(list(annotation_dict.values())[idx].split(',')[0])

    return gt_names, pred_names


def tensor_to_image(tensor):
    tensor = tensor.cpu().numpy()    
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def text_on_image(list_labels, image):    
    image = tensor_to_image(image)    
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 25)
    y = [0, 50, 100]  # y positions 
    for i in range(len(list_labels)):
        draw = ImageDraw.Draw(image)
        draw.text((0,y[i]),list_labels[i] , (0,0,255), font=font)

    image = transforms.ToTensor()(image)      
    return image




def evaluation_metrics(p,output, label): 
    metrics = {}
    for task in p['task_list']:
        
        if (task == 'class_scene') or (task == 'class_object'):            
            m = {}
            assert(output[task].shape == label[task].shape)                    
            gt = torch.zeros(label[task].shape)
            pred = torch.zeros(output[task].shape)
            # since label and output both are probabilities, 
            # convert then into a multi-label output by taking the top-5 prob = 1
            for i in range(label[task].shape[0]):
                idx_gt = torch.topk(label[task][i],3).indices
                idx_pred = torch.topk(output[task][i],3).indices
                gt[i][idx_gt]=1                     
                pred[i][idx_pred]=1   
            
            f1 = f1_score(gt, pred, average=None, zero_division = 1)          
            m['f1'] = f1.mean()  
            # score = precision_score(gt, pred, average=None, zero_division = 1)
            # m['precision'] = score.mean()
            # score = recall_score(gt, pred, average=None, zero_division = 1)
            # m['recall'] = score.mean()         
            metrics[task] = m


        elif task == 'segmentsemantic':
            from utils.seg_metrics import calculate_IoU , Seg_cross_entropy_metric
            
            m = {}   
            pred = output[task]
            new_dim = pred.shape[-2:] 
            out = label[task] 
            gt, _, _ =  torch.chunk(out, 3, dim = 1) 
            gt = F.interpolate(gt, size=new_dim)   
            IOU = calculate_IoU()  
            CE = Seg_cross_entropy_metric(p)
            m['IoU'] = IOU(p, gt, pred) 
            m['CrossEntropy'] = CE(pred, gt)    # ingore index 0 i.e. bg  
            metrics[task] = m  
             

        elif task =='depth_euclidean':
            from utils.depth_metrics import rmse,mean_abs_error, l1, get_depth_metric     
            m = {}  
            
            # pred = output[task].cuda()
            # gt = label[task].cuda()           

            if p['dataset_name'] == 'NYU':
                binary_mask = (torch.sum(label[task], dim=1) > 3 * 1e-5).unsqueeze(1).cuda()  

            elif p['dataset_name'] == 'Taskonomy':
                # label= label[task]*255
                binary_mask = (label[task] != 1).cuda()  

            else:
                print('task not found')
            
            if p['dataset_name'] == 'NYU':
                m['mae'], _, _ , _ , m['rmse'], _ = get_depth_metric(output[task].cuda(), label[task].cuda(), binary_mask)            
                metrics[task] = m  
            else:
                m['mae'], _, _ , _ , m['rmse'], _  = get_depth_metric(output[task].cuda(), label[task].cuda(), binary_mask)            
                metrics[task] = m 

            

        elif task =='surface_normal':
            from utils.metrics import sn_metrics
            m = {}
            
            gt = label[task]
            new_shape = gt.shape[-2:]
            pred = F.interpolate(output[task], size= new_shape, mode="bilinear") 
            m['cos_similarity'], angles = sn_metrics(pred, gt)

            if p['dataset_name'] == 'NYU':
                m['Angle Mean'] = np.mean(angles)
                m['Angle Median'] = np.median(angles)
                m['Angle RMSE'] = np.sqrt(np.mean(angles ** 2))
                m['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
                m['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
                m['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
                m['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100
            
            metrics[task] = m 

        elif task == 'edge_texture':
            from utils.metrics import edge_metrics
            m = {}
            gt = label[task].detach().cpu()
            pred = output[task].detach().cpu()           
            new_shape = pred.shape[-2:]            
            gt = F.interpolate(gt.type(torch.DoubleTensor), size= new_shape, mode="bilinear") 
            m['abs_err'] = edge_metrics(pred, gt)
            metrics[task] = m 

        else:
            print('unknown task for metric calculation')
    
    return metrics



def tb_logger(p,epoch,loss,metric,writer,set):


    for k,v in loss.items():
        writer.add_scalar(k+'/'+ set +'/loss', v, epoch)

    for task, v in metric.items():
        for m, val in v.items():
            writer.add_scalar(task+'/'+set+'/'+ m, val, epoch)




def create_config(config_exp):
    with open(config_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    return config



def draw_segmentation_map_taskonomy(outputs):

    label_map = [
               (255,255,255), # background
               #(255, 64, 192), # uncertain 
               (128, 0, 0), # bottle
               (0, 128, 0), # chair
               (128, 128, 0), # couch
               (0, 0, 128), # potted_plant
               (128, 0, 128), # bed
               (0, 128, 128), # dining_table 
               (128, 128, 128), # toilet
               (64, 0, 0), # tv
               (192, 0, 0), # microwave
               (64, 128, 0), # oven
               (192, 128, 0), # toaster
               (64, 0, 128), # sink
               (192, 0, 128), # refrigerator
               (64, 128, 128), # book
               (192, 128, 128), #clock
               (0, 64, 0), # vase                
            ]
    # labels = torch.argmax(outputs, dim=0).detach().cpu().numpy()    
    labels = outputs.detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return torch.tensor(segmented_image)



#NYU_CATEGORY_NAMES = ['otherprop', 'wall', 'floor', 'cabinet', 'bed', 'chair',
                    #   'sofa', 'table', 'door', 'window', 'bookshelf',
                    #   'picture', 'counter', 'blinds', 'desk', 'shelves',
                    #   'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
                    #   'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                    #   'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                    #   'person', 'night stand', 'toilet', 'sink', 'lamp',
                    #   'bathtub', 'bag', 'otherstructure', 'otherfurniture']

def draw_segmentation_map_NYU(outputs):
    
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
    labels = outputs.detach().cpu().numpy()    
    # labels = torch.argmax(outputs, dim=0).detach().cpu().numpy() 
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return torch.tensor(segmented_image)








def draw_binary_segmentation_map(p,outputs):
    outputs = outputs.squeeze(0)
    outputs = outputs.detach().cpu().numpy()  
    
    label_map = [               
               (255,255,255), # background
               (128, 0, 0), # other                      
            ]
    labels = outputs > p['threshold']*1
    # print(labels.shape)
    
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return torch.tensor(segmented_image)




class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out





class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



def get_combinations(p):
    ''' if there are 4 tasks this function will return 2^4 - 4 combinations'''
    tasks = p['task_list']
    from itertools import combinations
    Comb = []
    for i in range(2,len(tasks)+1):
        C = combinations(tasks,i)
        for list_C in C:
            Comb.append(list_C )
    return Comb


