import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder 
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: out}


class MultiTaskModel(nn.Module):
   
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        
        shared_representation = self.backbone(x)
        return {task: self.decoders[task](shared_representation) for task in self.tasks}

    # def get_last_shared_layer(self):
    #     return self.backbone.layer4


class MetaMultiTaskModel(nn.Module):
   
    def __init__(self, base_model: nn.Module, heads: nn.ModuleDict, tasks: list):
        super(MetaMultiTaskModel, self).__init__()
        # assert(set(decoders.keys()) == set(tasks))        
        # self.decoders = decoders
        self.backbone = base_model.backbone
        self.task_heads = base_model.decoders    
        for params in heads.parameters():
            params.requires_grad == True    
        self.task_heads.update(heads)    
        self.tasks = tasks

    def forward(self, x):        

        shared_representation = self.backbone(x)     
        return {task: self.task_heads[task](shared_representation) for task in self.tasks}