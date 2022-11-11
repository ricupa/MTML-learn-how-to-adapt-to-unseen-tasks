import torch
import torch.nn as nn


class ResNet_backbone(nn.Module):
    def __init__(self, original_model):
        super(ResNet_backbone, self).__init__()
        # self.original_model = original_model
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetb7_backbone(nn.Module):
    def __init__(self, original_model):
        super(EfficientNetb7_backbone, self).__init__()        
        self.in_features = original_model.classifier[1].in_features
        self.features =  nn.Sequential(*list(original_model.children())[:-2])  
        self.adaptive_layer =  nn.AdaptiveAvgPool2d(output_size=8)
        self.conv_layer = nn.Conv2d(self.in_features,2048,1)


    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_layer(x)
        x = self.conv_layer(x)
        return x
