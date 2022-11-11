import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from all_model.resnet import resnet50 
from utils.utils_common import *

class classification_head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1024, 3)  #in_channels = 2048
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(1024, 512, 4)  
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(512*6*6, 1000)
        self.dropout1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(1000, 500)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(500, num_classes)
        


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) 
        x = self.pool2(F.relu(self.conv2(x)))    
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)   #torch.sigmoid
        # x = self.act(x)
        return x


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
       

#     def forward(self, x):      
        
#         return self.double_conv(x)

# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv1 = DoubleConv(in_channels, out_channels, in_channels // 2)
        

#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv1(x)

#         return x, x.shape[1]



class segmentation_head_aspp(nn.Module):
    def __init__(self,p,in_channels,num_classes):
        super(segmentation_head_aspp, self).__init__()
        from all_model.ASPP import DeepLabHead
        self.num_classes = num_classes 
        self.in_channels = in_channels 
        self.aspp = DeepLabHead(in_channels = self.in_channels,num_classes=self.num_classes)
        self.tasks = p['task_list']
        

    def forward(self, x):       
        output = self.aspp(x) 
        #output = nn.functional.interpolate(output, size=(256,256), mode="bilinear") # (shape: (batch_size, num_classes, h, w))       
        
        return output
        

# class Depth_est_head(nn.Module):
#     def __init__(self, Encoder, num_features, block_channel):

#         super(model, self).__init__()
#         import all_model.depth_modules as modules
#         self.E = Encoder
#         self.D = modules.D(num_features)
#         self.MFF = modules.MFF(block_channel)
#         self.R = modules.R(block_channel)


#     def forward(self, x):
#         x_block1, x_block2, x_block3, x_block4 = self.E(x)
#         x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
#         x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
#         out = self.R(torch.cat((x_decoder, x_mff), 1))

#         return out






