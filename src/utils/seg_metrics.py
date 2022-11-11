
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np



class Seg_cross_entropy_metric(Module):
    """
    This function returns cross entropy error for semantic segmentation
    """

    def __init__(self, p):
        super(Seg_cross_entropy_metric, self).__init__()
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


    def forward(self, out, label):
        with torch.no_grad():
            label = label.permute(0,2,3,1)
            label = torch.reshape(label, (-1,))
            num_class = out.shape[1]
            mask = label< num_class
            label = label[mask].int()
            logits = out.permute(0, 2, 3, 1).contiguous().view(-1,num_class)
            logits = logits[mask]
            
            err = self.criterion(logits, label.long())
            
            return err.cpu().numpy()

class calculate_IoU(Module):

    def __init__(self):
        super(calculate_IoU, self).__init__()

    def forward(self,p, gt, pred):
        with torch.no_grad():
            eps=1e-8
            n_classes =  pred.shape[1]            

            pred = F.softmax(pred, dim = 1)
            pred = torch.argmax(pred,dim = 1)   
            iou =[]

            for gt, pred in zip(gt, pred):
                # jac = torch.zeros(n_classes)
                jac = 0
                tp, fp, fn = 0,0,0
                valid = (gt != 0)

                for i_part in range(n_classes):          
                    tmp_gt = (gt == i_part)
                    tmp_pred = (pred == i_part)          
                    tp += torch.sum((torch.tensor(tmp_gt & tmp_pred & valid)*1).view(-1),dtype = torch.float32) 
                    fp += torch.sum((torch.tensor(~tmp_gt & tmp_pred & valid)*1).view(-1),dtype = torch.float32)
                    fn += torch.sum((torch.tensor(tmp_gt & ~tmp_pred & valid)*1).view(-1),dtype = torch.float32)
                        
                jac = tp / max(tp + fp + fn, eps)
                
            
                iou.append(jac)

            iou = torch.tensor(iou) 
            iou = torch.mean(iou)     # for all the images in the batch
            return iou.cpu().numpy()