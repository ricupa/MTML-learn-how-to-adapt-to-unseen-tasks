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
from termcolor import colored


def add_task_heads_to_model(model, p):
    from utils.utils_common import get_head
    from all_model.models import MetaMultiTaskModel

    
    backbone_channels = 2048  # for resnet-50
    heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p['add_new_task_list']})
    task_list = p['add_new_task_list'] + p['task_list']
    
    newmodel = MetaMultiTaskModel(model, heads, task_list)
    
    return newmodel

