import torch
import torch.nn as nn
import torch.nn.functional as F
from all_model.resnet import resnet50
import os
from all_model.aspp import ASPP, ASPP_Bottleneck
from all_model.resnet import resnet50 
# from input_configs import p 
from utils.utils_common import *

class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()
        self.num_classes = 20              
        self.backbone, self.backbone_channels = get_backbone(p)       
        
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]
        
        feature_map = self.backbone(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
        
        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        print(output.shape)
        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

