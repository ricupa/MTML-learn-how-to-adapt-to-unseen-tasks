import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
from torch.autograd import Function

class Depth_combined_loss(nn.Module):
    ''' this is a combination of depth loss, gradient loss and normal loss
    as described in https://arxiv.org/pdf/1803.08673.pdf '''
    def __init__(self, weight=None, size_average=True):
        super(Depth_combined_loss, self).__init__()
        # self.weights = nn.Parameter(torch.zeros((3)))

    def forward(self, output, gt):

        '''
        output : model output
        depth : ground truth
        '''

        new_shape = output.shape[-2:]
        output = output[:,0,:,:].unsqueeze(1).cuda()        

        depth = F.interpolate(gt.float(), size=new_shape)
        depth = depth[:,0,:,:].unsqueeze(1).cuda()

        from utils.utils_common import Sobel 
        cos = nn.CosineSimilarity(dim=1, eps=0)

        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        
        get_grad = Sobel().cuda()

        depth_grad = get_grad(depth)
        output_grad = get_grad(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()

        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        
        
        return loss


class softmax_cross_entropy_with_softtarget(nn.Module):
    ''' SoftCrossEntropyLoss(input, target) = KLDivLoss(LogSoftmax(input), target)'''
    def __init__(self):
        super(softmax_cross_entropy_with_softtarget, self).__init__()
        

    def forward(self, input, target):

        log_probs = F.log_softmax(input)
        loss_kl = torch.nn.KLDivLoss(reduction='mean')  #mean, batchmean, none
        return loss_kl(log_probs, target)





class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """
    def __init__(self):
        super(DepthLoss, self).__init__()        
        self.loss = nn.L1Loss( reduction='mean')

    def forward(self, out, label):
        # label = label*255
        mask = (label != 1)
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))






class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, pred, label):            
        loss = torch.sqrt( torch.mean( torch.abs(torch.log(label)-torch.log(pred)) ** 2 ))
        return loss




class surface_normal_loss(nn.Module):
    def __init__(self):
        super(surface_normal_loss, self).__init__()
        self.cosine_similiarity = nn.CosineSimilarity()

    def forward(self, pred, gt): 
        
        new_shape = pred.shape[-2:]        
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)        
        gt = F.interpolate(gt.float(), size=new_shape)
        gt = gt.permute(0,2,3,1).contiguous().view(-1, 3)        
        labels = (gt.max(dim=1)[0] < 1)        
        pred = pred[labels]
        gt = gt[labels]
        pred = F.normalize(pred)
        gt = F.normalize(gt)        
        loss = 1 - self.cosine_similiarity(pred, gt)
        return loss.mean()




class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss, self).__init__()
        # self.l1_loss = nn.L1Loss()
        self.huberloss = nn.HuberLoss(reduction='mean', delta=0.5)

    def forward(self, pred, gt): 
        new_shape = pred.shape[-2:]
        # pred = torch.sigmoid(pred)
        gt = F.interpolate(gt.float(), size=new_shape)
        # loss = self.l1_loss(pred,gt)
        loss = self.huberloss(pred,gt)
        return loss





        






