import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
from torch.autograd import Function



class Seg_cross_entropy_loss(Module):
    """
    This function returns cross entropy error for semantic segmentation
    """

    def __init__(self, p):
        super(Seg_cross_entropy_loss, self).__init__()
        # self.softmax = nn.LogSoftmax(dim=1)

        if p['dataset_name'] == 'Taskonomy':
            num_classes = 17
            # weight = torch.zeros(num_classes)
            self.wt_file = "../dataset/taskonomy_dataset/utilities/semseg_prior_factor.npy"
            weight= torch.from_numpy(np.load(self.wt_file)).float()

        elif p['dataset_name'] == 'NYU':
            num_classes = 41
            weight = torch.ones(num_classes)*1.0
            weight[0] = 0.1
        else:
            print('Not Implemented error')

        weight=weight.cuda()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255)         


    def forward(self, pred, label):
        seg_num_class = pred.shape[1]
        new_shape = pred.shape[-2:]
        
        prediction = pred.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)        
        # batch_size = pred.shape[0] 
        label, _ , _ = torch.chunk(label, 3, dim = 1)  
        gt = F.interpolate(label.float(), size=new_shape)
        gt = gt.permute(0, 2, 3, 1).contiguous().view(-1)
        
        loss = self.criterion(prediction, gt.long())        
        return loss

