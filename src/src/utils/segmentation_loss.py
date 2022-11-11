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
            self.wt_file = "/proj/ltu_mtl/dataset/taskonomy_dataset/utilities/semseg_prior_factor.npy"
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


# class FocalBinaryTverskyLoss(Function):
    
#     @staticmethod
#     def forward(ctx, input, target):
#         _alpha = 0.5
#         _beta = 0.5
#         _gamma = 1.0
#         _epsilon = 1e-6
#         _reduction = 'mean'

#         batch_size = input.size(0)
#         _, input_label = input.max(1)

#         input_label = input_label.float()
#         target_label = target.float()

#         ctx.save_for_backward(input, target_label)

#         input_label = input_label.view(batch_size, -1)
#         target_label = target_label.view(batch_size, -1)

#         ctx.P_G = torch.sum(input_label * target_label, 1)  # TP
#         ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)  # FP
#         ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)  # FN

#         index = ctx.P_G / (ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon)
#         loss = torch.pow((1 - index), 1 / _gamma)
#         # target_area = torch.sum(target_label, 1)
#         # loss[target_area == 0] = 0
#         if _reduction == 'none':
#             loss = loss
#         elif _reduction == 'sum':
#             loss = torch.sum(loss)
#         else:
#             loss = torch.mean(loss)
#         return loss

#     # @staticmethod
#     def backward(ctx, grad_out):
#         """
#         :param ctx:
#         :param grad_out:
#         :return:
#         d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
#         (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
#                         = 2*P_G
#         (dT_loss/d_p0)=
#         """
#         _alpha = 0.5
#         _beta = 0.5
#         _gamma = 1.0
#         _reduction = 'mean'
#         _epsilon = 1e-6

#         inputs, target = ctx.saved_tensors
#         inputs = inputs.float()
#         target = target.float()
#         batch_size = inputs.size(0)
#         sum = ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon
#         P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
#         if inputs.dim() == 5:
#             sum = sum.view(batch_size, 1, 1, 1, 1)
#         elif inputs.dim() == 4:
#             sum = sum.view(batch_size, 1, 1, 1)
#             P_G = ctx.P_G.view(batch_size, 1, 1, 1)
#         sub = (_alpha * (1 - target) + target) * P_G

#         dL_dT = (1 / _gamma) * torch.pow((P_G / sum), (1 / _gamma - 1))
#         dT_dp0 = -2 * (target / sum - sub / sum / sum)
#         dL_dp0 = dL_dT * dT_dp0

#         dT_dp1 = _beta * (1 - target) * P_G / sum / sum
#         dL_dp1 = dL_dT * dT_dp1
#         grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
#         # grad_input = torch.cat((grad_out.item() * dL_dp0, dL_dp0 * grad_out.item()), dim=1)
#         return grad_input, None





# class MultiTverskyLoss(nn.Module):
#     """
#     Tversky Loss for segmentation adaptive with multi class segmentation
#     """

#     def __init__(self,p, alpha, beta, gamma, weights=None):
#         """
#         :param alpha (Tensor, float, optional): controls the penalty for false positives.
#         :param beta (Tensor, float, optional): controls the penalty for false negative.
#         :param gamma (Tensor, float, optional): focal coefficient
#         :param weights (Tensor, optional): a manual rescaling weight given to each
#             class. If given, it has to be a Tensor of size `C`
#         """
#         super(MultiTverskyLoss, self).__init__()
#         self.dataset = p['dataset_name']
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.weights = weights

#     def forward(self, inputs, targets):       
#         new_shape = inputs.shape[-2:]
#         num_class = inputs.size(1)
#         inputs = F.softmax(inputs, dim = 1)
#         imputs = torch.argmax(inputs, dim = 1)
#         gt, _ , _ = torch.chunk(targets, 3, dim = 1)
        
#         targets = F.interpolate(gt.float(), size=new_shape)
#         targets = targets.long()
        
#         if self.weights is not None:
#             assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
#             weights = self.weights
#         else:
#             if self.dataset == 'Taskonomy':
#                 file = "../taskonomy_dataset/utilities/semseg_prior_factor.npy"
#                 self.weights = torch.from_numpy(np.load(file)).float()
#             else:
#                 temp = torch.bincount(targets.view(-1), minlength = num_class)  
#                 weights = [1/val if val > 0 else torch.tensor(0.0) for val in temp]    
#                 weights[0] = 0 ## i.e.background        
#                 weights = torch.tensor(weights)                
#                 assert len(weights) == num_class
#             # weights = [1.0 / num_class] * num_class      
#             #       
#         input_slices = torch.split(inputs, [1] * num_class, dim=1)
#         weight_losses = 0.0
#         for idx in range(num_class):
#             input_idx = input_slices[idx]
#             input_idx = torch.cat((1 - input_idx, input_idx), dim=1)
#             target_idx = (targets == idx) * 1
#             loss_func = FocalBinaryTverskyLoss(self.alpha, self.beta, self.gamma)
#             loss_idx = loss_func.apply(input_idx, target_idx)
#             weight_losses += loss_idx * weights[idx]
#         # loss = torch.Tensor(weight_losses)
#         # loss = loss.to(inputs.device)
#         # loss = torch.sum(loss)
#         return weight_losses




# class MultiTverskyLoss(nn.Module):
#     """
#     Tversky Loss for segmentation adaptive with multi class segmentation
#     """

#     def __init__(self, p, alpha, beta, gamma, weights=None):
#         super(MultiTverskyLoss, self).__init__()
#         self.dataset = p['dataset_name']
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma


#     def forward(self, true, logits):
#         """Computes the Tversky loss [1].
#         Args:
#             true: a tensor of shape [B, H, W] or [B, 1, H, W].
#             logits: a tensor of shape [B, C, H, W]. Corresponds to
#                 the raw output or logits of the model.
#             alpha: controls the penalty for false positives.
#             beta: controls the penalty for false negatives.
#             eps: added to the denominator for numerical stability.
#         Returns:
#             tversky_loss: the Tversky loss.
#         Notes:
#             alpha = beta = 0.5 => dice coeff
#             alpha = beta = 1 => tanimoto coeff
#             alpha + beta = 1 => F beta coeff
#         References:
#             [1]: https://arxiv.org/abs/1706.05721
#         """
#         eps=1e-7
#         num_classes = logits.shape[1]
#         # if num_classes == 1:
#         #     true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
#         #     true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         #     true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         #     true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         #     true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         #     pos_prob = torch.sigmoid(logits)
#         #     neg_prob = 1 - pos_prob
#         #     probas = torch.cat([pos_prob, neg_prob], dim=1)
#         # else:
#         true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(logits, dim=1)
#         true_1_hot = true_1_hot.type(logits.type())
#         dims = (0,) + tuple(range(2, true.ndimension()))
#         intersection = torch.sum(probas * true_1_hot, dims)
#         fps = torch.sum(probas * (1 - true_1_hot), dims)
#         fns = torch.sum((1 - probas) * true_1_hot, dims)
#         num = intersection
#         denom = intersection + (self.alpha * fps) + (self.beta * fns)
#         tversky_loss = (num / (denom + eps)).mean()
#         return (1 - tversky_loss)